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
    APPROVED_STATUSES, LIVE_COLLECTION, PLATFORM_COLLECTION, get_db, get_search_manager,
)

log = logging.getLogger("recommender")

TOP_N = 20
THRESHOLD = 0.1
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


class _Candidates:
    """Candidate vectors by id, kept across users in one run (the same internships come up again and again)."""

    def __init__(self, db):
        self.db = db
        self.cache: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {}

    def get(self, ids: Iterable[str]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        from bson import ObjectId

        ids = list(ids)
        missing = [i for i in ids if i not in self.cache]
        if missing:
            oids = [ObjectId(i) for i in missing]
            proj = {"tfidf_vector": 1, "bert_vector": 1}
            found: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            live = {"_id": {"$in": oids}, "$or": [{"isActive": True}, {"isActive": {"$exists": False}}]}
            plat = {**_visible_platform_filter(), "_id": {"$in": oids}}
            for col, q in ((LIVE_COLLECTION, live), (PLATFORM_COLLECTION, plat)):
                for d in self.db[col].find(q, proj):
                    if isinstance(d.get("bert_vector"), list) and isinstance(d.get("tfidf_vector"), list):
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
        {"$or": [{"isActive": True}, {"isActive": {"$exists": False}}], "bert_vector": {"$type": "array"}}, {"_id": 1})]
    ids += [str(d["_id"]) for d in db[PLATFORM_COLLECTION].find(
        {**_visible_platform_filter(), "bert_vector": {"$type": "array"}}, {"_id": 1})]
    return ids


def score_user(db, mgr, cands: _Candidates, tfidf: List[float], bert: List[float]) -> List[Tuple[str, float]]:
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
            scored.append((i, score))
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
    mgr = get_search_manager(db, dim=len(bert))
    recs = score_user(db, mgr, _Candidates(db), tfidf, bert)
    save_recommendations(db, user_id, recs)
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

    mgr = None
    cands = _Candidates(db)
    for user in db["users"].find(query, {"resume.tfidf_vector": 1, "resume.bert_vector": 1}):
        stats["users"] += 1
        try:
            bert = user["resume"]["bert_vector"]
            if mgr is None:
                mgr = get_search_manager(db, dim=len(bert))
            recs = score_user(db, mgr, cands, user["resume"]["tfidf_vector"], bert)
            save_recommendations(db, user["_id"], recs)
            stats["updated"] += 1
        except Exception as err:
            log.error("Recommendation failed for user %s: %s", user.get("_id"), err)
            stats["errors"] += 1
    log.info("Recommended for %s", stats)
    return stats


def run_fanout(user_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    recommend_users() with coalescing: when one is already running, ask it to go once more when it finishes
    instead of piling up (each approval in a moderator batch would otherwise start its own pass).
    A targeted run (user_ids) is never coalesced.
    """
    global _fanout_dirty
    if user_ids is not None:
        return recommend_users(user_ids)
    if not _FANOUT_LOCK.acquire(blocking=False):
        _fanout_dirty = True
        return {"status": "coalesced"}
    try:
        while True:
            _fanout_dirty = False
            stats = recommend_users()
            if not _fanout_dirty:
                return stats
    finally:
        _FANOUT_LOCK.release()
