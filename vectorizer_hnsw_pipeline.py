"""
vectorizer_hnsw_pipeline.py
────────────────────────────
Internship Vectorization & HNSW Graph Maintenance Pipeline.
Located inside the vectoriser service. Student recommendations live in recommender.py.

Flow (only APPROVED staged internships are processed):
  1. Fetch internships from collection('internships.mod-unvectorised') whose
     moderation.status is 'auto_approved' or 'manually_approved'
     (optionally restricted to specific _ids).
  2. Generate BERT (768-d) & TF-IDF (15000-d) embeddings.
  3. Insert vector node into the HNSW graph (node ID mapped to document _id).
  4. Move the vectorized document into collection('internships').
  5. Delete it from collection('internships.mod-unvectorised').
  6. Persist the updated graph into collection('internships.graph') ONCE per run.

Anything not approved (pending_review / auto_rejected / manually_rejected)
stays in 'internships.mod-unvectorised' untouched.

Employer-posted internships ('internships.this-platform') are vectorized IN PLACE and inserted into the same
graph (see run_platform_vectorizer_pipeline); the recommender resolves graph ids in both collections.

Also exposes graph search (`search_graph` / `get_search_manager`) for the recommender.
"""

import os
import sys
import json
import logging
import pickle
import threading
import functools
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pymongo
from dotenv import load_dotenv

# Ensure local module directory is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from task1_module import flatten, tfidf_transform, load_vectorizer
from task2_module import load_bert_model, bert_transform
from task3_internship_profiles_module import boost_tfidf_vector

# Native hnswlib when available; otherwise a brute-force numpy fallback with the same interface.
try:
    import hnswlib  # type: ignore
    HAS_NATIVE_HNSWLIB = True
except ImportError:
    HAS_NATIVE_HNSWLIB = False

log = logging.getLogger("vectorizer_hnsw_pipeline")

STAGING_COLLECTION = "internships.mod-unvectorised"
LIVE_COLLECTION = "internships"
PLATFORM_COLLECTION = "internships.this-platform"   # employer-posted internships (source "ifind")
GRAPH_COLLECTION = "internships.graph"
GRAPH_DOC_ID = "hnsw_internships_index"
APPROVED_STATUSES = ("auto_approved", "manually_approved")

# Serialises pipeline runs / graph persistence (single-worker uvicorn).
PIPELINE_LOCK = threading.Lock()

# Scraped internships whose application deadline has passed are moved here: out of 'internships', out of the graph.
EXPIRED_COLLECTION = "internships.expired"


def deadline_passed(value, now: Optional[datetime] = None) -> bool:
    """
    True when an application deadline is over. Scraped listings store it as a naive ISO string (UTC), employer posts as a Date.
    A date-only value (midnight) lasts until the end of that day, the same rule as ifind lib/utils.ts deadlineState().
    No / unreadable deadline = not expired.
    """
    if not value:
        return False
    try:
        d = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return False
    d = d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)
    if d.hour == d.minute == d.second == 0:
        d = d.replace(hour=23, minute=59, second=59)
    return d < (now or datetime.now(timezone.utc))


# ─────────────────────────────────────────────────────────────────────────────
# Fallback index (used only when native hnswlib isn't installed)
# ─────────────────────────────────────────────────────────────────────────────

class FallbackHNSWIndex:
    """
    Pure NumPy stand-in with the hnswlib.Index interface subset we use.
    ponytail: brute-force cosine scan, NOT a graph. Fine for <~10k vectors;
    install hnswlib (see requirements.txt) for real HNSW.
    """
    impl = "fallback"

    def __init__(self, space: str = 'cosine', dim: int = 768):
        self.space = space
        self.dim = dim
        self.M = 16
        self.ef_construction = 200
        self.ef = 50
        self.max_elements = 100000
        self.vectors: Dict[int, np.ndarray] = {}
        self.entry_point_id: Optional[int] = None

    def init_index(self, max_elements: int = 100000, M: int = 16, ef_construction: int = 200, random_seed: int = 100):
        self.max_elements = max_elements
        self.M = M
        self.ef_construction = ef_construction

    def set_ef(self, ef: int):
        self.ef = ef

    def add_items(self, data: np.ndarray, ids: List[int]):
        data_arr = np.atleast_2d(np.array(data, dtype=np.float32))
        id_list = [ids] if isinstance(ids, (int, np.integer)) else list(ids)
        for vec, item_id in zip(data_arr, id_list):
            norm = np.linalg.norm(vec)
            self.vectors[int(item_id)] = vec / norm if norm > 0 else vec
            if self.entry_point_id is None:
                self.entry_point_id = int(item_id)

    def get_current_count(self) -> int:
        return len(self.vectors)

    def knn_query(self, data: np.ndarray, k: int = 1):
        q = np.array(data, dtype=np.float32).reshape(-1)
        n = np.linalg.norm(q)
        q = q / n if n > 0 else q
        ids = list(self.vectors.keys())
        if not ids:
            return np.array([[]]), np.array([[]])
        sims = np.stack([self.vectors[i] for i in ids]) @ q
        top = np.argsort(-sims)[:k]
        return np.array([[ids[i] for i in top]]), np.array([[1.0 - float(sims[i]) for i in top]])

    def to_bytes(self) -> bytes:
        return pickle.dumps({
            "space": self.space, "dim": self.dim, "M": self.M,
            "ef_construction": self.ef_construction, "ef": self.ef,
            "max_elements": self.max_elements, "vectors": self.vectors,
            "entry_point_id": self.entry_point_id,
        })

    def load_bytes(self, blob: bytes):
        state = pickle.loads(blob)
        self.space = state.get("space", "cosine")
        self.dim = state.get("dim", 768)
        self.M = state.get("M", 16)
        self.ef_construction = state.get("ef_construction", 200)
        self.ef = state.get("ef", 50)
        self.max_elements = state.get("max_elements", 100000)
        self.vectors = state.get("vectors", {})
        self.entry_point_id = state.get("entry_point_id", None)


# ─────────────────────────────────────────────────────────────────────────────
# HNSW Index Manager (index creation, load, add, search, DB persistence)
# ─────────────────────────────────────────────────────────────────────────────

class HNSWIndexManager:
    """
    Manages the HNSW index over internship 768-d BERT vectors.
    Maps MongoDB _id (str) <-> integer node id and persists to 'internships.graph'.
    """
    def __init__(self, dim: int = 768, space: str = 'cosine', storage_dir: Optional[str] = None):
        self.dim = dim
        self.space = space
        self.storage_dir = Path(storage_dir) if storage_dir else SCRIPT_DIR / "data" / "hnsw"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_dir / "internships_index.bin"

        self.uid_to_id: Dict[str, int] = {}
        self.id_to_uid: Dict[str, str] = {}
        self.index = None
        self.loaded_updated_at: Optional[str] = None

    # -- construction ---------------------------------------------------------

    def _new_index(self, max_elements: int):
        if HAS_NATIVE_HNSWLIB:
            idx = hnswlib.Index(space=self.space, dim=self.dim)
            idx.init_index(max_elements=max_elements, M=16, ef_construction=200)
            idx.set_ef(50)
        else:
            idx = FallbackHNSWIndex(space=self.space, dim=self.dim)
            idx.init_index(max_elements=max_elements, M=16, ef_construction=200)
        return idx

    def _reset(self, max_elements: int = 100000):
        self.uid_to_id, self.id_to_uid = {}, {}
        self.index = self._new_index(max_elements)

    def initialize_or_load(self, db: pymongo.database.Database, max_elements: int = 100000):
        """
        Load the graph from 'internships.graph'. If it is missing, or was written
        by a different implementation (native vs fallback), rebuild it from the
        vectors stored on the live 'internships' collection.
        """
        saved = db[GRAPH_COLLECTION].find_one({"_id": GRAPH_DOC_ID})
        if not saved or "graph_data" not in saved:
            self._reset(max_elements)
            if db[LIVE_COLLECTION].estimated_document_count() or db[PLATFORM_COLLECTION].estimated_document_count():
                self.rebuild_from_db(db)
            return

        blob = bytes(saved["graph_data"])
        saved_impl = saved.get("impl") or ("fallback" if blob[:2] == b"\x80\x05" else "native")
        current_impl = "native" if HAS_NATIVE_HNSWLIB else "fallback"
        if saved_impl != current_impl:
            log.warning("Stored graph is '%s' but runtime is '%s' — rebuilding from DB vectors",
                        saved_impl, current_impl)
            self._reset(max_elements)
            self.rebuild_from_db(db)
            return

        self.uid_to_id = saved.get("uid_to_id", {})
        self.id_to_uid = saved.get("id_to_uid", {})
        count = len(self.uid_to_id)
        try:
            if current_impl == "native":
                self.index = hnswlib.Index(space=self.space, dim=self.dim)
                tmp = self.storage_dir / "temp_restore.bin"
                tmp.write_bytes(blob)
                self.index.load_index(str(tmp), max_elements=max(max_elements, count + 10000))
                self.index.set_ef(50)
            else:
                self.index = FallbackHNSWIndex(space=self.space, dim=self.dim)
                self.index.load_bytes(blob)
        except Exception as err:
            # e.g. written by another hnswlib build (chroma-hnswlib on a Windows machine vs hnswlib on the Space)
            log.warning("Stored graph could not be loaded (%s) — rebuilding from DB vectors", err)
            self._reset(max_elements)
            self.rebuild_from_db(db)
            return
        self.loaded_updated_at = saved.get("updatedAt")
        log.info("Restored %s HNSW graph from MongoDB (%d nodes)", current_impl, count)

    def rebuild_from_db(self, db: pymongo.database.Database, persist: bool = True) -> int:
        """Rebuild the whole graph from live internships that have a valid bert_vector."""
        self._reset()
        n = 0
        now = datetime.now(timezone.utc)
        cursor = db[LIVE_COLLECTION].find(
            {"bert_vector": {"$type": "array"}, "isActive": {"$ne": False}},
            {"bert_vector": 1, "deadlineDate": 1},
        )
        for doc in cursor:
            vec = doc["bert_vector"]
            if deadline_passed(doc.get("deadlineDate"), now):
                continue   # expired: never in the graph
            if len(vec) == self.dim:
                self.add_node(str(doc["_id"]), np.array(vec, dtype=np.float32))
                n += 1
        # Employer-posted internships: approved + vectorized, any non-draft status (visibility is checked at query time,
        # so a paused post that resumes is still in the graph)
        for doc in db[PLATFORM_COLLECTION].find(
            {"bert_vector": {"$type": "array"}, "isActive": {"$ne": False}, "bannedByCompany": {"$ne": True}, "status": {"$ne": "draft"},
             "moderation.status": {"$in": list(APPROVED_STATUSES)}},
            {"bert_vector": 1, "deadlineDate": 1},
        ):
            vec = doc["bert_vector"]
            if deadline_passed(doc.get("deadlineDate"), now):
                continue
            if len(vec) == self.dim:
                self.add_node(str(doc["_id"]), np.array(vec, dtype=np.float32))
                n += 1
        log.info("Rebuilt HNSW graph from DB: %d nodes", n)
        if persist:
            self.persist_graph(db)
        return n

    # -- mutation / query -----------------------------------------------------

    def add_node(self, doc_uid: str, vector: np.ndarray) -> int:
        doc_uid_str = str(doc_uid)
        if doc_uid_str in self.uid_to_id:
            int_id = self.uid_to_id[doc_uid_str]
        else:
            int_id = len(self.uid_to_id)
            self.uid_to_id[doc_uid_str] = int_id
            self.id_to_uid[str(int_id)] = doc_uid_str

        if HAS_NATIVE_HNSWLIB:
            if self.index.get_current_count() >= self.index.get_max_elements():
                self.index.resize_index(self.index.get_max_elements() * 2)

        self.index.add_items(np.array(vector, dtype=np.float32).reshape(1, -1), [int_id])
        return int_id

    def search(self, vector: List[float], k: int = 20) -> List[Tuple[str, float]]:
        """Top-k nearest internships as (doc_id, cosine_similarity), best first."""
        count = self.index.get_current_count() if self.index else 0
        if count == 0:
            return []
        q = np.array(vector, dtype=np.float32).reshape(1, -1)
        kk = min(k, count)
        while kk > 0:
            try:
                labels, dists = self.index.knn_query(q, k=kk)
                break
            except RuntimeError:  # native hnswlib: k exceeds reachable elements
                kk //= 2
        else:
            return []
        return [
            (self.id_to_uid[str(int(l))], round(1.0 - float(d), 6))
            for l, d in zip(labels[0], dists[0])
            if str(int(l)) in self.id_to_uid
        ]

    def persist_graph(self, db: pymongo.database.Database):
        """Persist graph bytes + id mappings into 'internships.graph'."""
        if HAS_NATIVE_HNSWLIB:
            self.index.save_index(str(self.index_file))
            graph_bytes = self.index_file.read_bytes()
        else:
            graph_bytes = self.index.to_bytes()

        self.loaded_updated_at = datetime.now(timezone.utc).isoformat()
        db[GRAPH_COLLECTION].replace_one(
            {"_id": GRAPH_DOC_ID},
            {
                "_id": GRAPH_DOC_ID,
                "graph_data": graph_bytes,
                "impl": "native" if HAS_NATIVE_HNSWLIB else "fallback",
                "uid_to_id": self.uid_to_id,
                "id_to_uid": self.id_to_uid,
                "count": len(self.uid_to_id),
                "dim": self.dim,
                "metric": self.space,
                "updatedAt": self.loaded_updated_at,
            },
            upsert=True,
        )
        log.info("HNSW graph persisted (%d nodes)", len(self.uid_to_id))


# ─────────────────────────────────────────────────────────────────────────────
# DB / model helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_db(mongo_uri: Optional[str] = None, db_name: Optional[str] = None) -> pymongo.database.Database:
    for env_path in [
        SCRIPT_DIR / ".env.local",
        SCRIPT_DIR / ".env",
        SCRIPT_DIR.parent.parent / "ifind" / ".env.local",
        SCRIPT_DIR.parent.parent / "internScraper" / ".env",
    ]:
        if env_path.exists():
            load_dotenv(env_path)

    uri = mongo_uri or os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI is not set")
    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=10000)
    return client[db_name or os.getenv("MONGODB_DB_NAME", "ifind")]


@functools.lru_cache(maxsize=1)
def load_recommender_models():
    """Load TF-IDF vectorizer, BERT SentenceTransformer model, and skill cluster dicts."""
    tfidf_path = SCRIPT_DIR / "task1-tfidf_vectorizer.pkl"
    clusters_path = SCRIPT_DIR / "skill_clusters.json"
    s2c_path = SCRIPT_DIR / "skill_to_category.json"

    log.info("Loading TF-IDF vectorizer from %s...", tfidf_path)
    tfidf_model = load_vectorizer(str(tfidf_path))

    log.info("Loading BERT model...")
    try:
        bert_model = load_bert_model("sentence-transformers/all-mpnet-base-v2")
    except Exception:
        bert_model = load_bert_model("all-MiniLM-L6-v2")

    clusters = json.loads(clusters_path.read_text(encoding="utf-8")) if clusters_path.exists() else {}
    s2c = json.loads(s2c_path.read_text(encoding="utf-8")) if s2c_path.exists() else {}
    return tfidf_model, bert_model, clusters, s2c


def encode_internship_text(
    text: str,
    tfidf_model,
    bert_model,
    clusters: dict,
    s2c: dict,
    boost: float = 0.15
) -> Tuple[List[float], List[float]]:
    """Generate boosted TF-IDF vector (15,000-d) & BERT vector (768-d)."""
    flat = flatten(text, mode="text")
    tfidf_base = tfidf_transform(text, tfidf_model, mode="text")

    if clusters and s2c:
        tfidf_boosted = boost_tfidf_vector(
            base_vector=tfidf_base,
            text=flat,
            vectorizer=tfidf_model,
            skill_clusters=clusters,
            skill_to_category=s2c,
            boost_weight=boost,
        )
    else:
        tfidf_boosted = tfidf_base

    bert_vec = bert_transform(text, bert_model, mode="text")
    return tfidf_boosted.tolist(), bert_vec.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_internship_vectorizer_hnsw_pipeline(
    mongo_uri: Optional[str] = None,
    db_name: Optional[str] = None,
    batch_limit: Optional[int] = None,
    ids: Optional[List[str]] = None,
    moderator_id: Optional[str] = None,
) -> dict:
    """
    Vectorize + index + publish approved staged internships.

    ids: restrict to these staging _ids (24-hex strings). Processes each doc one at a time:
         vectorizes, sets moderation.status to manually_approved, moves from staging to live
         collection, and persists the graph incrementally.
    Returns {processed, errors, skipped, indexed_ids}.
    """
    from bson import ObjectId

    stats: Dict[str, Any] = {"processed": 0, "errors": 0, "skipped": 0, "indexed_ids": []}

    with PIPELINE_LOCK:
        db = get_db(mongo_uri, db_name)
        source_col = db[STAGING_COLLECTION]
        target_col = db[LIVE_COLLECTION]

        query: Dict[str, Any] = {}
        if ids is not None:
            oids = []
            for i in ids:
                try:
                    oids.append(ObjectId(i))
                except Exception:
                    stats["skipped"] += 1
            query["_id"] = {"$in": oids}
            stats["skipped"] += max(0, len(oids) - source_col.count_documents(query))
        else:
            query["moderation.status"] = {"$in": list(APPROVED_STATUSES)}

        cursor = source_col.find(query)
        if batch_limit and batch_limit > 0:
            cursor = cursor.limit(batch_limit)
        docs = list(cursor)
        if not docs:
            log.info("Staging check: 0 approved internships waiting in '%s' (nothing to vectorise)", STAGING_COLLECTION)
            return stats

        log.info("Found %d approved staged internship(s) to vectorise and index into HNSW graph", len(docs))
        tfidf_model, bert_model, clusters, s2c = load_recommender_models()
        dim = getattr(bert_model, "get_sentence_embedding_dimension", lambda: 768)()

        mgr = HNSWIndexManager(dim=dim, space='cosine')
        mgr.initialize_or_load(db)

        for idx, doc in enumerate(docs, start=1):
            title = doc.get("name") or doc.get("title") or "Untitled"
            company = doc.get("company") or "Unknown"
            doc_id = str(doc.get("_id"))
            try:
                if deadline_passed(doc.get("deadlineDate")):
                    # approved too late: archive instead of vectorising / publishing
                    log.warn("[%d/%d] Archiving expired internship '%s' at '%s' (id=%s, deadline was %s)",
                             idx, len(docs), title, company, doc_id, doc.get("deadlineDate"))
                    db[EXPIRED_COLLECTION].replace_one({"_id": doc["_id"]}, {**doc, "expiredAt": datetime.now(timezone.utc)}, upsert=True)
                    source_col.delete_one({"_id": doc["_id"]})
                    stats["expired"] = stats.get("expired", 0) + 1
                    continue
                full_text = f"{title}. {company}. {doc.get('summary') or doc.get('description') or ''}".strip()
                tfidf_vec, bert_vec = encode_internship_text(full_text, tfidf_model, bert_model, clusters, s2c)
                doc["tfidf_vector"] = tfidf_vec
                doc["bert_vector"] = bert_vec

                # Set moderation status to manually_approved if not already approved
                if "moderation" not in doc or not isinstance(doc["moderation"], dict):
                    doc["moderation"] = {}
                if doc["moderation"].get("status") not in APPROVED_STATUSES:
                    doc["moderation"]["status"] = "manually_approved"
                    doc["moderation"]["reviewedAt"] = datetime.now(timezone.utc)
                    if moderator_id:
                        doc["moderation"]["reviewedBy"] = moderator_id
                    doc["moderation"]["rejectionReason"] = None
                elif moderator_id and not doc["moderation"].get("reviewedBy"):
                    doc["moderation"]["reviewedBy"] = moderator_id

                mgr.add_node(doc_id, np.array(bert_vec, dtype=np.float32))
                target_col.replace_one({"_id": doc["_id"]}, doc, upsert=True)
                source_col.delete_one({"_id": doc["_id"]})
                mgr.persist_graph(db)

                stats["processed"] += 1
                stats["indexed_ids"].append(doc_id)
                log.info("[%d/%d] Vectorised & published: '%s' at '%s' (id=%s)",
                         idx, len(docs), title, company, doc_id)
            except Exception as err:
                log.error("[%d/%d] Error vectorising '%s' (id=%s): %s", idx, len(docs), title, doc_id, err)
                stats["errors"] += 1

        if stats["processed"]:
            mgr.persist_graph(db)

    if stats["errors"] == 0:
        log.ok("Staged vectorisation complete: %d/%d vectorised & moved to live (skipped=%d, expired=%d, graph now %d nodes)",
               stats["processed"], len(docs), stats["skipped"], stats.get("expired", 0), len(mgr.uid_to_id))
    else:
        log.warn("Staged vectorisation finished with errors: %d vectorised, %d failed out of %d internships (skipped=%d, expired=%d, graph now %d nodes)",
                 stats["processed"], stats["errors"], len(docs), stats["skipped"], stats.get("expired", 0), len(mgr.uid_to_id))
    return stats


def run_platform_vectorizer_pipeline(
    ids: List[str],
    mongo_uri: Optional[str] = None,
    db_name: Optional[str] = None,
) -> dict:
    """
    Vectorize approved employer-posted internships IN PLACE in 'internships.this-platform' and add them to the
    HNSW graph.

    Unlike the scraper flow there is no staging and no move: the document already lives in its final
    collection (the employer owns its lifecycle), so we write `tfidf_vector` / `bert_vector` / `vectorizedAt`
    onto it and insert its id into the graph. Only moderation.status in (auto_approved, manually_approved) is
    processed; `ids` is required (24-hex strings) so this can never sweep the whole collection.

    Returns {processed, errors, skipped, vectorized_ids}.
    """
    from bson import ObjectId

    stats: Dict[str, Any] = {"processed": 0, "errors": 0, "skipped": 0, "vectorized_ids": []}
    oids = []
    for i in ids or []:
        try:
            oids.append(ObjectId(i))
        except Exception:
            stats["skipped"] += 1
    if not oids:
        log.info("Platform vectoriser: 0 valid IDs provided, nothing to process")
        return stats

    with PIPELINE_LOCK:
        db = get_db(mongo_uri, db_name)
        col = db[PLATFORM_COLLECTION]
        docs = list(col.find({"_id": {"$in": oids}, "bannedByCompany": {"$ne": True}, "moderation.status": {"$in": list(APPROVED_STATUSES)}}))
        stats["skipped"] += len(oids) - len(docs)
        if not docs:
            log.info("Platform vectoriser: none of the %d provided IDs are approved employer internships in '%s'",
                     len(oids), PLATFORM_COLLECTION)
            return stats

        log.info("Found %d approved employer internship(s) in '%s' to vectorise in place",
                 len(docs), PLATFORM_COLLECTION)
        tfidf_model, bert_model, clusters, s2c = load_recommender_models()
        dim = getattr(bert_model, "get_sentence_embedding_dimension", lambda: 768)()
        mgr = HNSWIndexManager(dim=dim, space='cosine')
        mgr.initialize_or_load(db)

        for idx, doc in enumerate(docs, start=1):
            title = doc.get("name") or "Untitled"
            company = doc.get("company") or "Unknown"
            doc_id = str(doc.get("_id"))
            try:
                text = f"{title}. {company}. {doc.get('summary') or ''}".strip()
                tfidf_vec, bert_vec = encode_internship_text(text, tfidf_model, bert_model, clusters, s2c)
                col.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"tfidf_vector": tfidf_vec, "bert_vector": bert_vec, "vectorizedAt": datetime.now(timezone.utc)}},
                )
                mgr.add_node(doc_id, np.array(bert_vec, dtype=np.float32))
                stats["processed"] += 1
                stats["vectorized_ids"].append(doc_id)
                log.info("[%d/%d] Vectorised employer internship: '%s' at '%s' (id=%s)",
                         idx, len(docs), title, company, doc_id)
            except Exception as err:
                log.error("[%d/%d] Error vectorising platform internship '%s' (id=%s): %s",
                          idx, len(docs), title, doc_id, err)
                stats["errors"] += 1

        if stats["processed"]:
            mgr.persist_graph(db)

    if stats["errors"] == 0:
        log.ok("Platform vectorisation complete: %d/%d vectorised in place (skipped=%d, graph now %d nodes)",
               stats["processed"], len(docs), stats["skipped"], len(mgr.uid_to_id))
    else:
        log.warn("Platform vectorisation finished with errors: %d vectorised, %d failed out of %d internships (skipped=%d)",
                 stats["processed"], stats["errors"], len(docs), stats["skipped"])
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Expiry sweep
# ─────────────────────────────────────────────────────────────────────────────

def purge_expired(db=None) -> Dict[str, Any]:
    """
    Remove expired internships from everything the recommender can see.
      scraped ('internships'):           moved to 'internships.expired' (kept for applied / saved history)
      employer-posted (this-platform):   stay where they are (the employer owns them), only dropped from the graph
    The graph is rebuilt (without expired nodes) when it held any of them. Idempotent; safe to call often.
    """
    db = db if db is not None else get_db()
    now = datetime.now(timezone.utc)
    stats: Dict[str, Any] = {"moved": 0, "platform_expired": 0, "graph_rebuilt": False}
    with PIPELINE_LOCK:
        gone: List[str] = []
        for d in db[LIVE_COLLECTION].find({"deadlineDate": {"$ne": None}}, {"deadlineDate": 1}):
            if deadline_passed(d.get("deadlineDate"), now):
                full = db[LIVE_COLLECTION].find_one({"_id": d["_id"]})
                if full is None:
                    continue
                full.pop("tfidf_vector", None)
                full.pop("bert_vector", None)   # vectors are useless once it can no longer be recommended
                db[EXPIRED_COLLECTION].replace_one({"_id": d["_id"]}, {**full, "expiredAt": now}, upsert=True)
                db[LIVE_COLLECTION].delete_one({"_id": d["_id"]})
                gone.append(str(d["_id"]))
        stats["moved"] = len(gone)
        for d in db[PLATFORM_COLLECTION].find({"deadlineDate": {"$ne": None}}, {"deadlineDate": 1}):
            if deadline_passed(d.get("deadlineDate"), now):
                gone.append(str(d["_id"]))
                stats["platform_expired"] += 1
        mgr = HNSWIndexManager()
        mgr.initialize_or_load(db)
        # also catches nodes of listings another service (internScraper) already moved to the expired collection
        from bson import ObjectId
        oids = [ObjectId(u) for u in mgr.uid_to_id if ObjectId.is_valid(u)]
        gone += [str(d["_id"]) for d in db[EXPIRED_COLLECTION].find({"_id": {"$in": oids}}, {"_id": 1})] if oids else []
        if any(i in mgr.uid_to_id for i in gone):
            mgr.rebuild_from_db(db)
            stats["graph_rebuilt"] = True
    log.info("Expiry sweep: %s", stats)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Graph search (used by the recommender)
# ─────────────────────────────────────────────────────────────────────────────

_SEARCH_MGR: Optional[HNSWIndexManager] = None


def get_search_manager(db=None, dim: int = 768) -> HNSWIndexManager:
    """The persisted graph, loaded once and reloaded whenever the stored copy has changed."""
    global _SEARCH_MGR
    db = db if db is not None else get_db()
    meta = db[GRAPH_COLLECTION].find_one({"_id": GRAPH_DOC_ID}, {"updatedAt": 1})
    stored = meta.get("updatedAt") if meta else None
    if _SEARCH_MGR is None or _SEARCH_MGR.loaded_updated_at != stored:
        mgr = HNSWIndexManager(dim=dim)
        mgr.initialize_or_load(db)
        _SEARCH_MGR = mgr
    return _SEARCH_MGR


def search_graph(vector: List[float], k: int = 20) -> List[Tuple[str, float]]:
    """Top-k nearest internships (scraped and employer-posted) as (id, cosine similarity)."""
    return get_search_manager(dim=len(vector)).search(vector, k)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_internship_vectorizer_hnsw_pipeline()
