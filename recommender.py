"""
recommender.py
──────────────
Student recommendations, moved here from ifind/lib/recommendation so everything that touches vectors and the
HNSW graph lives in one service.

For one student: run their BERT vector through the WHOLE graph (k-NN), fetch the hit internships (scraped
'internships' and employer-posted 'internships.this-platform'), score each one with the weighted cosine
(0.4 TF-IDF + 0.6 BERT, vectors are L2-normalised so the dot product is the cosine), drop scores under the
threshold, keep the top N and save them on the user as `recommendedInternships`.

Triggered by (see app.py):
  - /encode-resume with user_id + recommend   a student's resume or profile changed
  - /vectorize-hnsw, /vectorize-platform      new internships entered the graph -> every student is re-run
  - /recommend-all                            admin refresh (all students, or the given user_ids)
"""

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from vectorizer_hnsw_pipeline import (
    APPROVED_STATUSES, LIVE_COLLECTION, PLATFORM_COLLECTION, deadline_passed, get_db, get_search_manager, purge_expired,
)

log = logging.getLogger("recommender")

TOP_N = 20
THRESHOLD = 0.1
# Small nudges for the student's location (the profile "preferences" city / state): they only reorder close matches.
CITY_BOOST = 0.08
STATE_BOOST = 0.04
REMOTE_BOOST = 0.03
TFIDF_WEIGHT = 0.4
BERT_WEIGHT = 0.6
K = TOP_N * 2 * 2   # candidates pulled from the graph (same as the old HNSWStrategy: topN*2, doubled)

# One fan-out at a time; requests that arrive meanwhile are folded into one extra pass at the end.
_FANOUT_LOCK = threading.Lock()
_fanout_dirty = False


def _visible_platform_filter() -> Dict[str, Any]:
    """Same rule as ifind lib/employer/visibility.ts visibleToStudentsFilter()."""
    now = datetime.now(timezone.utc)
    return {
        "status": "published",
        "moderation.status": {"$in": list(APPROVED_STATUSES)},
        "isActive": {"$ne": False},
        "bannedByCompany": {"$ne": True},   # company banned by a moderator (ifind companies.restriction.status == "banned")
        "$or": [{"deadlineDate": None}, {"deadlineDate": {"$gt": now}}],
    }


def _norm(v) -> str:
    return str(v or "").strip().lower()


def location_boost(location: Optional[Dict[str, Any]], is_remote: bool, addresses: Optional[list]) -> float:
    """Boost for an internship that is remote, or in the student's city / state. 0 when the student has no location."""
    if not location:
        return 0.0
    city, state = _norm(location.get("city")), _norm(location.get("state"))
    if not city and not state:
        return 0.0
    boost = REMOTE_BOOST if is_remote else 0.0
    for a in addresses or []:
        if not isinstance(a, dict):
            continue
        if city and _norm(a.get("city")) == city:
            boost = max(boost, CITY_BOOST)
        elif state and _norm(a.get("state")) == state:
            boost = max(boost, STATE_BOOST)
    return boost


class _Candidates:
    """Candidate vectors by id, kept across users in one run (the same internships come up again and again)."""

    def __init__(self, db):
        self.db = db
        self.cache: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {}
        self.where: Dict[str, Tuple[bool, list]] = {}   # id -> (isRemote, address[]) for the location boost

    def get(self, ids: Iterable[str]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        from bson import ObjectId

        ids = list(ids)
        missing = [i for i in ids if i not in self.cache]
        if missing:
            oids = [ObjectId(i) for i in missing]
            proj = {"tfidf_vector": 1, "bert_vector": 1, "isRemote": 1, "address": 1, "deadlineDate": 1}
            now = datetime.now(timezone.utc)
            found: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            live = {"_id": {"$in": oids}, "$or": [{"isActive": True}, {"isActive": {"$exists": False}}]}
            plat = {**_visible_platform_filter(), "_id": {"$in": oids}}
            for col, q in ((LIVE_COLLECTION, live), (PLATFORM_COLLECTION, plat)):
                for d in self.db[col].find(q, proj):
                    if deadline_passed(d.get("deadlineDate"), now):
                        continue   # expired: never recommended, whatever the graph still holds
                    if isinstance(d.get("bert_vector"), list) and isinstance(d.get("tfidf_vector"), list):
                        self.where[str(d["_id"])] = (bool(d.get("isRemote")), d.get("address") if isinstance(d.get("address"), list) else [])
                        found[str(d["_id"])] = (
                            np.asarray(d["tfidf_vector"], dtype=np.float32),
                            np.asarray(d["bert_vector"], dtype=np.float32),
                        )
            for i in missing:
                self.cache[i] = found.get(i)   # None = not recommendable right now (inactive, paused, no vectors)
        return {i: self.cache[i] for i in ids if self.cache.get(i) is not None}


def _brute_force_ids(db) -> List[str]:
    """Graph unavailable: every recommendable internship (the old BruteForceStrategy). ponytail: O(n), fine as a fallback."""
    ids = [str(d["_id"]) for d in db[LIVE_COLLECTION].find(
        {"$or": [{"isActive": True}, {"isActive": {"$exists": False}}], "bert_vector": {"$type": "array"}}, {"_id": 1, "deadlineDate": 1})
        if not deadline_passed(d.get("deadlineDate"))]
    ids += [str(d["_id"]) for d in db[PLATFORM_COLLECTION].find(
        {**_visible_platform_filter(), "bert_vector": {"$type": "array"}}, {"_id": 1, "deadlineDate": 1})
        if not deadline_passed(d.get("deadlineDate"))]
    return ids


def score_user(db, mgr, cands: _Candidates, tfidf: List[float], bert: List[float], location: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
    """Run one student's vectors through the graph and return the top N (id, score), best first."""
    try:
        ids = [i for i, _ in mgr.search(bert, K)]
    except Exception as err:
        log.error("Graph search failed, falling back to a full scan: %s", err)
        ids = _brute_force_ids(db)

    u_tfidf = np.asarray(tfidf, dtype=np.float32)
    u_bert = np.asarray(bert, dtype=np.float32)
    scored = []
    for i, (c_tfidf, c_bert) in cands.get(ids).items():
        t = float(u_tfidf @ c_tfidf) if c_tfidf.shape == u_tfidf.shape else 0.0
        b = float(u_bert @ c_bert) if c_bert.shape == u_bert.shape else 0.0
        score = t * TFIDF_WEIGHT + b * BERT_WEIGHT
        if score >= THRESHOLD:
            remote, addresses = cands.where.get(i, (False, []))
            scored.append((i, score + location_boost(location, remote, addresses)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:TOP_N]


def save_recommendations(db, user_id, recs: List[Tuple[str, float]]) -> None:
    from bson import ObjectId

    db["users"].update_one(
        {"_id": ObjectId(str(user_id))},
        {"$set": {
            "recommendedInternships.updatedAt": datetime.now(timezone.utc),
            "recommendedInternships.recommendedList": [ObjectId(i) for i, _ in recs],
            "recommendedInternships.recommendedScores": [{"id": ObjectId(i), "score": round(s, 3)} for i, s in recs],
        }},
    )


def recommend_user(user_id: str, tfidf: List[float], bert: List[float], db=None) -> int:
    """Run ONE student through the whole graph and save their list. Returns how many were recommended."""
    db = db if db is not None else get_db()
    from bson import ObjectId

    mgr = get_search_manager(db, dim=len(bert))
    graph_count = mgr.index.get_current_count() if mgr and mgr.index else len(mgr.uid_to_id)
    user_doc = db["users"].find_one({"_id": ObjectId(str(user_id))}, {"username": 1, "email": 1, "name": 1, "city": 1, "state": 1})
    username = user_doc.get("username") or user_doc.get("email") or user_doc.get("name") or str(user_id) if user_doc else str(user_id)

    log.info("Running recommender for student '%s' (id=%s) against %d graph internships...",
             username, user_id, graph_count)
    recs = score_user(db, mgr, _Candidates(db), tfidf, bert, user_doc)
    save_recommendations(db, user_id, recs)
    top_score = round(recs[0][1], 3) if recs else 0.0
    log.ok("Recommended %d internships for student '%s' (top match: %.3f)", len(recs), username, top_score)
    return len(recs)


def recommend_users(user_ids: Optional[List[str]] = None, db=None) -> Dict[str, int]:
    """Run every student with resume vectors (or just `user_ids`) through the whole graph."""
    from bson import ObjectId

    db = db if db is not None else get_db()
    stats = {"users": 0, "updated": 0, "errors": 0}
    query: Dict[str, Any] = {"resume.bert_vector": {"$type": "array"}, "resume.tfidf_vector": {"$type": "array"}}
    if user_ids is not None:
        oids = []
        for u in user_ids:
            try:
                oids.append(ObjectId(u))
            except Exception:
                stats["errors"] += 1
        query["_id"] = {"$in": oids}

    mgr = get_search_manager(db)
    graph_count = mgr.index.get_current_count() if mgr and mgr.index else (len(mgr.uid_to_id) if mgr else 0)
    total_users = db["users"].count_documents(query)

    target_desc = f"{total_users} specified student(s)" if user_ids is not None else f"every student ({total_users} with active resumes)"
    log.info("Running recommender fan-out on %s against %d graph internships (top %d per student)...",
             target_desc, graph_count, TOP_N)

    if total_users == 0:
        log.info("No matching students found in database — 0 recommendations needed")
        return stats

    cands = _Candidates(db)
    cursor = db["users"].find(
        query,
        {"username": 1, "email": 1, "name": 1, "resume.tfidf_vector": 1, "resume.bert_vector": 1, "city": 1, "state": 1}
    )

    idx = 0
    for user in cursor:
        idx += 1
        stats["users"] += 1
        username = user.get("username") or user.get("email") or user.get("name") or str(user["_id"])
        try:
            bert = user["resume"]["bert_vector"]
            if mgr is None or mgr.dim != len(bert):
                mgr = get_search_manager(db, dim=len(bert))
            recs = score_user(db, mgr, cands, user["resume"]["tfidf_vector"], bert, user)
            save_recommendations(db, user["_id"], recs)
            stats["updated"] += 1
            top_score = round(recs[0][1], 3) if recs else 0.0
            log.info("[%d/%d] Recommended for student '%s': %d internships matched (top score: %.3f)",
                     idx, total_users, username, len(recs), top_score)
        except Exception as err:
            log.error("[%d/%d] Recommendation failed for student '%s' (id=%s): %s",
                      idx, total_users, username, user.get("_id"), err)
            stats["errors"] += 1

    if stats["errors"] == 0:
        log.ok("Recommender complete: %d/%d students updated successfully, 0 failed (graph size: %d internships)",
               stats["updated"], stats["users"], graph_count)
    else:
        log.warn("Recommender finished with issues: %d updated, %d failed out of %d students",
                 stats["updated"], stats["errors"], stats["users"])
    return stats


def run_fanout(user_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    recommend_users() with coalescing: when one is already running, ask it to go once more when it finishes
    instead of piling up (each approval in a moderator batch would otherwise start its own pass).
    A targeted run (user_ids) is never coalesced.
    """
    global _fanout_dirty
    try:
        purge_expired()   # expired internships leave the collection and the graph before anyone is scored
    except Exception as err:
        log.error("Expiry sweep failed (the recommender still skips expired internships): %s", err)
    if user_ids is not None:
        return recommend_users(user_ids)
    if not _FANOUT_LOCK.acquire(blocking=False):
        _fanout_dirty = True
        log.info("Recommender fan-out already in progress — marked dirty to re-run on completion")
        return {"status": "coalesced"}
    try:
        pass_count = 1
        while True:
            _fanout_dirty = False
            log.info("Starting recommender fan-out pass #%d on all students...", pass_count)
            stats = recommend_users()
            if not _fanout_dirty:
                return stats
            pass_count += 1
            log.info("Fan-out marked dirty during pass #%d, running subsequent pass...", pass_count - 1)
    finally:
        _FANOUT_LOCK.release()
