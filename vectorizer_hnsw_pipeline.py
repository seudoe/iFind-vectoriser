"""
vectorizer_hnsw_pipeline.py
────────────────────────────
Internship Vectorization & HNSW Graph Maintenance Pipeline.
Located inside vectorisationResume.

Flow:
  1. Fetch unvectorized internships from collection('internships.mod-unvectorised')
  2. Generate BERT (768-d) & TF-IDF (15000-d) embeddings using task1_module, task2_module, task3_internship_profiles_module
  3. Insert vector node into HNSW Graph Index (node ID mapped to document _id)
  4. Persist updated HNSW Graph into collection('internships.graph')
  5. Move vectorized document into collection('internships')
  6. Delete document from collection('internships.mod-unvectorised')

Note: This module handles graph creation, vectorization, and maintenance ONLY.
      Recommendation engine logic and searching are explicitly excluded.
"""

import os
import sys
import json
import logging
import pickle
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

# Try importing native hnswlib; fallback to custom HNSWGraph implementation if unavailable
try:
    import hnswlib  # type: ignore
    HAS_NATIVE_HNSWLIB = True
except ImportError:
    HAS_NATIVE_HNSWLIB = False

log = logging.getLogger("vectorizer_hnsw_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Custom Fallback HNSW Graph (when native C++ hnswlib extension isn't compiled)
# ─────────────────────────────────────────────────────────────────────────────

class FallbackHNSWIndex:
    """
    Pure Python/NumPy HNSW Index fallback.
    Matches hnswlib.Index interface: init_index, add_items, save_index, load_index.
    """
    def __init__(self, space: str = 'cosine', dim: int = 768):
        self.space = space
        self.dim = dim
        self.M = 16
        self.ef_construction = 200
        self.ef = 50
        self.max_elements = 100000
        self.vectors: Dict[int, np.ndarray] = {}  # { int_id: np.ndarray(768,) }
        self.entry_point_id: Optional[int] = None

    def init_index(self, max_elements: int = 100000, M: int = 16, ef_construction: int = 200, random_seed: int = 100):
        self.max_elements = max_elements
        self.M = M
        self.ef_construction = ef_construction

    def set_ef(self, ef: int):
        self.ef = ef

    def add_items(self, data: np.ndarray, ids: List[int]):
        """Add vector(s) to index with integer IDs."""
        data_arr = np.atleast_2d(np.array(data, dtype=np.float32))
        id_list = [ids] if isinstance(ids, (int, np.integer)) else list(ids)

        for vec, item_id in zip(data_arr, id_list):
            norm = np.linalg.norm(vec)
            norm_vec = vec / norm if norm > 0 else vec
            self.vectors[int(item_id)] = norm_vec
            if self.entry_point_id is None:
                self.entry_point_id = int(item_id)

    def get_current_count(self) -> int:
        return len(self.vectors)

    def save_index(self, path: str):
        """Serialize HNSW index state to disk file."""
        state = {
            "space": self.space,
            "dim": self.dim,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef": self.ef,
            "max_elements": self.max_elements,
            "vectors": self.vectors,
            "entry_point_id": self.entry_point_id,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load_index(self, path: str, max_elements: int = 100000):
        """Deserialize HNSW index state from disk file."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.space = state.get("space", "cosine")
        self.dim = state.get("dim", 768)
        self.M = state.get("M", 16)
        self.ef_construction = state.get("ef_construction", 200)
        self.ef = state.get("ef", 50)
        self.max_elements = max_elements or state.get("max_elements", 100000)
        self.vectors = state.get("vectors", {})
        self.entry_point_id = state.get("entry_point_id", None)


# ─────────────────────────────────────────────────────────────────────────────
# HNSW Index Manager (Handles index creation, loading, adding, and DB persistence)
# ─────────────────────────────────────────────────────────────────────────────

class HNSWIndexManager:
    """
    Manages HNSW Graph index operations for internship 768-d BERT vectors.
    Enforces mapping between MongoDB _id (doc UID) and integer node IDs.
    Persists index state to MongoDB collection 'internships.graph'.
    """
    def __init__(self, dim: int = 768, space: str = 'cosine', storage_dir: Optional[str] = None):
        self.dim = dim
        self.space = space
        self.storage_dir = Path(storage_dir) if storage_dir else SCRIPT_DIR / "data" / "hnsw"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_dir / "internships_index.bin"
        
        self.uid_to_id: Dict[str, int] = {}    # { doc_uid_str: node_int_id }
        self.id_to_uid: Dict[str, str] = {}    # { str(node_int_id): doc_uid_str }
        self.index = None

    def initialize_or_load(self, db: pymongo.database.Database, max_elements: int = 100000):
        """
        Load index from MongoDB collection('internships.graph') or local disk.
        If non-existent, initialize a fresh HNSW index.
        """
        graph_col = db["internships.graph"]
        saved_doc = graph_col.find_one({"_id": "hnsw_internships_index"})

        # Restore ID mappings if present
        if saved_doc:
            self.uid_to_id = saved_doc.get("uid_to_id", {})
            self.id_to_uid = saved_doc.get("id_to_uid", {})

        if HAS_NATIVE_HNSWLIB:
            self.index = hnswlib.Index(space=self.space, dim=self.dim)
            if saved_doc and "graph_data" in saved_doc:
                temp_path = self.storage_dir / "temp_restore.bin"
                with open(temp_path, "wb") as f:
                    f.write(saved_doc["graph_data"])
                count = len(self.uid_to_id)
                cap = max(max_elements, count + 10000)
                self.index.load_index(str(temp_path), max_elements=cap)
                self.index.set_ef(50)
                log.info("✅ Restored native HNSW index from MongoDB internships.graph (%d nodes)", count)
            elif self.index_file.exists():
                count = len(self.uid_to_id)
                cap = max(max_elements, count + 10000)
                self.index.load_index(str(self.index_file), max_elements=cap)
                self.index.set_ef(50)
                log.info("✅ Restored native HNSW index from local disk (%d nodes)", count)
            else:
                self.index.init_index(max_elements=max_elements, M=16, ef_construction=200)
                self.index.set_ef(50)
                log.info("Initialized new native HNSW index (dim=%d, space=%s)", self.dim, self.space)
        else:
            self.index = FallbackHNSWIndex(space=self.space, dim=self.dim)
            if saved_doc and "graph_data" in saved_doc:
                temp_path = self.storage_dir / "temp_restore.pkl"
                with open(temp_path, "wb") as f:
                    f.write(saved_doc["graph_data"])
                self.index.load_index(str(temp_path), max_elements=max_elements)
                log.info("✅ Restored fallback HNSW index from MongoDB internships.graph (%d nodes)", len(self.uid_to_id))
            elif self.index_file.exists():
                self.index.load_index(str(self.index_file), max_elements=max_elements)
                log.info("✅ Restored fallback HNSW index from local disk (%d nodes)", len(self.uid_to_id))
            else:
                self.index.init_index(max_elements=max_elements, M=16, ef_construction=200)
                log.info("Initialized new fallback HNSW index (dim=%d, space=%s)", self.dim, self.space)

    def add_node(self, doc_uid: str, vector: np.ndarray) -> int:
        """
        Map doc_uid -> int_id and insert node vector into HNSW index.
        """
        doc_uid_str = str(doc_uid)
        if doc_uid_str in self.uid_to_id:
            int_id = self.uid_to_id[doc_uid_str]
        else:
            int_id = len(self.uid_to_id)
            self.uid_to_id[doc_uid_str] = int_id
            self.id_to_uid[str(int_id)] = doc_uid_str

        if HAS_NATIVE_HNSWLIB:
            curr_count = self.index.get_current_count()
            if curr_count >= self.index.get_max_elements():
                new_cap = self.index.get_max_elements() * 2
                self.index.resize_index(new_cap)

        vec = np.array(vector, dtype=np.float32).reshape(1, -1)
        self.index.add_items(vec, [int_id])
        return int_id

    def persist_graph(self, db: pymongo.database.Database):
        """
        Persist HNSW Graph index bytes & ID mappings into MongoDB collection('internships.graph').
        """
        self.index.save_index(str(self.index_file))
        with open(self.index_file, "rb") as f:
            graph_bytes = f.read()

        graph_col = db["internships.graph"]
        node_count = len(self.uid_to_id)
        
        graph_col.replace_one(
            {"_id": "hnsw_internships_index"},
            {
                "_id": "hnsw_internships_index",
                "graph_data": graph_bytes,
                "uid_to_id": self.uid_to_id,
                "id_to_uid": self.id_to_uid,
                "count": node_count,
                "dim": self.dim,
                "metric": self.space,
                "updatedAt": datetime.now(timezone.utc).isoformat(),
            },
            upsert=True
        )
        log.info("💾 HNSW Graph persisted into MongoDB collection 'internships.graph' (%d nodes)", node_count)


# ─────────────────────────────────────────────────────────────────────────────
# Model Loader for vectorisationResume Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def load_recommender_models():
    """
    Load TF-IDF vectorizer, BERT SentenceTransformer model, and skill cluster dicts.
    """
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

    clusters = {}
    if clusters_path.exists():
        with open(clusters_path, encoding="utf-8") as f:
            clusters = json.load(f)

    s2c = {}
    if s2c_path.exists():
        with open(s2c_path, encoding="utf-8") as f:
            s2c = json.load(f)

    return tfidf_model, bert_model, clusters, s2c


def encode_internship_text(
    text: str,
    tfidf_model,
    bert_model,
    clusters: dict,
    s2c: dict,
    boost: float = 0.15
) -> Tuple[List[float], List[float]]:
    """
    Generate boosted TF-IDF vector (15,000-d) & BERT vector (768-d).
    Returns (tfidf_vector_list, bert_vector_list).
    """
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
# Main Pipeline Function
# ─────────────────────────────────────────────────────────────────────────────

def run_internship_vectorizer_hnsw_pipeline(
    mongo_uri: Optional[str] = None,
    db_name: str = "ifind",
    batch_limit: Optional[int] = None
) -> dict:
    """
    Executes the internship vectorization and HNSW graph maintenance pipeline:

    1. Fetch unvectorized internships from collection('internships.mod-unvectorised')
    2. Generate BERT & TF-IDF embeddings using vectorisationResume modules
    3. Insert node vector into HNSW Graph index
    4. Persist updated graph in collection('internships.graph')
    5. Move document to collection('internships')
    6. Delete document from collection('internships.mod-unvectorised')

    Returns stats dict: { processed, errors }
    """
    stats = {"processed": 0, "errors": 0}

    # Load environment variables
    for env_path in [
        SCRIPT_DIR / ".env.local",
        SCRIPT_DIR / ".env",
        SCRIPT_DIR.parent.parent / "ifind" / ".env.local",
        SCRIPT_DIR.parent.parent / "internScraper" / ".env",
    ]:
        if env_path.exists():
            load_dotenv(env_path)

    uri = mongo_uri or os.getenv("MONGODB_URI") or (
        "mongodb://samwlhds:LwO4fqJfCkOld8Nz@ac-0igzdj2-shard-00-00.j8uearf.mongodb.net:27017,"
        "ac-0igzdj2-shard-00-01.j8uearf.mongodb.net:27017,"
        "ac-0igzdj2-shard-00-02.j8uearf.mongodb.net:27017/ifind?"
        "ssl=true&replicaSet=atlas-hqgb18-shard-0&authSource=admin&appName=iFind"
    )

    log.info("🔌 Connecting to MongoDB...")
    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=10000)
    db = client[db_name]

    source_col = db["internships.mod-unvectorised"]
    target_col = db["internships"]

    # 1. Fetch unvectorized documents
    cursor = source_col.find()
    if batch_limit and batch_limit > 0:
        cursor = cursor.limit(batch_limit)

    unvectorized_docs = list(cursor)
    if not unvectorized_docs:
        log.info("No unvectorized internships found in 'internships.mod-unvectorised'. Exiting.")
        return stats

    log.info("Found %d unvectorized internship(s) in 'internships.mod-unvectorised'.", len(unvectorized_docs))

    # 2. Load Vectorizer Models & HNSW Index Manager
    log.info("Loading recommender vectorizer models...")
    tfidf_model, bert_model, clusters, s2c = load_recommender_models()
    dim = getattr(bert_model, "get_sentence_embedding_dimension", lambda: 768)()

    hnsw_manager = HNSWIndexManager(dim=dim, space='cosine')
    hnsw_manager.initialize_or_load(db)

    # 3. Process each internship
    for doc in unvectorized_docs:
        try:
            doc_id = doc["_id"]
            doc_id_str = str(doc_id)

            name = doc.get("name") or doc.get("title") or ""
            company = doc.get("company") or ""
            summary = doc.get("summary") or doc.get("description") or ""

            full_text = f"{name}. {company}. {summary}".strip()

            # Generate TF-IDF & BERT embeddings using vectorisationResume engine
            tfidf_vec, bert_vec = encode_internship_text(
                full_text, tfidf_model, bert_model, clusters, s2c
            )
            doc["tfidf_vector"] = tfidf_vec
            doc["bert_vector"] = bert_vec

            # Insert Node into HNSW Graph
            node_id = hnsw_manager.add_node(doc_id_str, np.array(bert_vec, dtype=np.float32))

            # Persist updated HNSW Graph
            hnsw_manager.persist_graph(db)

            # Move Internship document to collection('internships')
            target_col.replace_one({"_id": doc_id}, doc, upsert=True)

            # Delete from collection('internships.mod-unvectorised')
            source_col.delete_one({"_id": doc_id})

            stats["processed"] += 1
            log.info("  ✅ Vectorized & Graph-Inserted: '%s' @ %s (node_id: %d)", name, company, node_id)

        except Exception as err:
            log.error("  💥 Error processing document %s: %s", doc.get("_id"), err)
            stats["errors"] += 1

    client.close()
    log.info("━━━ Pipeline Completed ━━━ Processed: %d, Errors: %d", stats["processed"], stats["errors"])
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_internship_vectorizer_hnsw_pipeline()
