"""
app.py — Hybrid Internship Recommender API
──────────────────────────────────────────
Two endpoints only:

  POST /encode-internships  — batch encode internship descriptions → vectors
  POST /encode-resume       — encode one user's resume → vectors

Vectors are returned to Next.js which stores them in MongoDB.
All cosine similarity matching happens on the Next.js side (no API call needed).

Deploy: Hugging Face Spaces → Docker → port 7860
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

sys.path.insert(0, os.path.dirname(__file__))

from task1_module import flatten, tfidf_transform, load_vectorizer
from task2_module import load_bert_model, bert_transform, bert_batch_transform
from task3_internship_profiles_module import boost_tfidf_vector

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Global model state — loaded ONCE at startup, reused for every request
# ─────────────────────────────────────────────────────────────────────────────

MODELS: Dict[str, Any] = {}

MAX_BATCH = 70   # hard cap — reject anything larger


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading models …")
    try:
        import json

        tfidf_path    = os.getenv("TFIDF_PATH",    "task1-tfidf_vectorizer.pkl")
        bert_path     = os.getenv("BERT_PATH",     "sentence-transformers/all-mpnet-base-v2")
        clusters_path = os.getenv("CLUSTERS_PATH", "skill_clusters.json")
        s2c_path      = os.getenv("S2C_PATH",      "skill_to_category.json")

        MODELS["tfidf"] = load_vectorizer(tfidf_path)
        log.info(f"  TF-IDF loaded  vocab={len(MODELS['tfidf'].vocabulary_):,}")

        MODELS["bert"] = load_bert_model(bert_path)
        log.info("  BERT loaded")

        with open(clusters_path) as f:
            MODELS["clusters"] = json.load(f)
        with open(s2c_path) as f:
            MODELS["s2c"] = json.load(f)
        log.info(f"  Clusters loaded  {len(MODELS['clusters'])} domains")

        MODELS["ready"] = True
        log.info("All models ready ✅")

    except Exception as e:
        log.error(f"Startup failed: {e}")
        MODELS["ready"] = False
        MODELS["error"] = str(e)

    yield
    MODELS.clear()


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title    = "iFind Recommender — Vector Encoder",
    version  = "2.0.0",
    lifespan = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # lock to your Next.js domain in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


def _check_ready():
    if not MODELS.get("ready"):
        raise HTTPException(
            status_code = 503,
            detail      = MODELS.get("error", "Models still loading, try again shortly"),
        )


def _encode_text(text: str, boost: float = 0.15) -> Dict[str, List[float]]:
    """
    Flatten → boosted TF-IDF vector + BERT vector.
    Both are L2-normalised. Returns plain Python lists (JSON-serialisable).
    """
    flat = flatten(text, mode="text")

    tfidf_base   = tfidf_transform(text, MODELS["tfidf"], mode="text")
    tfidf_boosted = boost_tfidf_vector(
        base_vector       = tfidf_base,
        text              = flat,
        vectorizer        = MODELS["tfidf"],
        skill_clusters    = MODELS["clusters"],
        skill_to_category = MODELS["s2c"],
        boost_weight      = boost,
    )
    bert_vec = bert_transform(text, MODELS["bert"], mode="text")

    return {
        "tfidf": tfidf_boosted.tolist(),   # list of 15,000 floats
        "bert":  bert_vec.tolist(),         # list of 768 floats
    }


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class InternshipItem(BaseModel):
    """
    Accepts either the clean API shape OR the raw MongoDB internship document.

    Clean shape:   { id, title, description }
    MongoDB shape: { _id: {$oid:...} | str, name, summary, ... }

    Extra fields are silently ignored.
    """
    model_config = {"extra": "ignore", "populate_by_name": True}

    id:          str | None = Field(None)
    title:       str | None = Field(None)
    description: str | None = Field(None)
    name:        str | None = Field(None)
    summary:     str | None = Field(None)

    # _id comes in as { "$oid": "..." } or a plain string — resolved in validator
    _resolved_id: str = ""

    @model_validator(mode="before")
    @classmethod
    def extract_mongo_id(cls, data: Any) -> Any:
        if isinstance(data, dict):
            raw_id = data.get("_id")
            if raw_id is not None:
                if isinstance(raw_id, dict):
                    # { "$oid": "664a..." }
                    data["id"] = raw_id.get("$oid", str(raw_id))
                else:
                    data["id"] = str(raw_id)
        return data

    def resolved_id(self) -> str:
        return self.id or "unknown"

    def resolved_title(self) -> str:
        return self.title or self.name or "Untitled"

    def resolved_description(self) -> str:
        return self.description or self.summary or ""


class EncodeInternshipsRequest(BaseModel):
    internships:  List[InternshipItem] = Field(
        ...,
        description=f"Max {MAX_BATCH} internships per call. Accepts clean {{id,title,description}} or raw MongoDB documents.",
    )
    boost_weight: float = Field(0.15, description="Skill cluster boost magnitude")


class InternshipVectors(BaseModel):
    id:    str
    tfidf: List[float]   # 15,000 floats
    bert:  List[float]   # 768 floats


class EncodeInternshipsResponse(BaseModel):
    encoded:  int
    vectors:  List[InternshipVectors]


# Resume can be plain text or the parsed JSON object from OpenAI/Gemini
ResumeInput = Union[str, Dict[str, Any]]


class EncodeResumeRequest(BaseModel):
    resume:       ResumeInput = Field(
        ...,
        description="Plain text string OR the parsedData JSON from your resume parser",
    )
    boost_weight: float = Field(0.15)


class EncodeResumeResponse(BaseModel):
    tfidf: List[float]   # 15,000 floats
    bert:  List[float]   # 768 floats


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
def root():
    return {
        "status":  "ready" if MODELS.get("ready") else "loading",
        "service": "iFind Vector Encoder",
        "endpoints": ["/encode-internships", "/encode-resume"],
    }


@app.get("/health", tags=["health"])
def health():
    return {
        "ready": MODELS.get("ready", False),
        "error": MODELS.get("error"),
    }


@app.post(
    "/encode-internships",
    response_model = EncodeInternshipsResponse,
    tags           = ["encode"],
    summary        = "Batch encode internship descriptions → tfidf + bert vectors",
)
def encode_internships(req: EncodeInternshipsRequest):
    """
    Send up to **70 internships** per call.
    Returns `{ id, tfidf: [...15000], bert: [...768] }` for each.

    Store the returned vectors in MongoDB:
    ```
    internship.tfidf_vector = result.tfidf
    internship.bert_vector  = result.bert
    ```

    Call this:
    - When your scraper adds new internships (pass only the new ones)
    - When an internship description is edited
    - Once on first deploy to index existing internships
    """
    _check_ready()

    if len(req.internships) > MAX_BATCH:
        raise HTTPException(
            status_code = 422,
            detail      = f"Batch too large: {len(req.internships)} > max {MAX_BATCH}. Split into smaller batches.",
        )

    results: List[InternshipVectors] = []

    for item in req.internships:
        # Combine title + description for richer signal
        full_text = f"{item.resolved_title()}. {item.resolved_description()}"
        try:
            vecs = _encode_text(full_text, boost=req.boost_weight)
            results.append(InternshipVectors(
                id    = item.resolved_id(),
                tfidf = vecs["tfidf"],
                bert  = vecs["bert"],
            ))
        except Exception as e:
            log.error(f"Failed to encode internship {item.resolved_id()}: {e}")
            raise HTTPException(status_code=500, detail=f"Encoding failed for id={item.resolved_id()}: {e}")

    log.info(f"Encoded {len(results)} internship(s)")
    return EncodeInternshipsResponse(encoded=len(results), vectors=results)


@app.post(
    "/encode-resume",
    response_model = EncodeResumeResponse,
    tags           = ["encode"],
    summary        = "Encode one user's resume → tfidf + bert vectors",
)
def encode_resume(req: EncodeResumeRequest):
    """
    Send one user's resume (plain text or parsed JSON).
    Returns `{ tfidf: [...15000], bert: [...768] }`.

    Store the returned vectors in MongoDB:
    ```
    user.resume.tfidf_vector = result.tfidf
    user.resume.bert_vector  = result.bert
    ```

    Call this:
    - When a user uploads a resume
    - When a user clicks "Re-extract"
    """
    _check_ready()

    # If resume is a dict (parsed JSON), flatten it to text first
    if isinstance(req.resume, dict):
        resume_text = flatten(req.resume, mode="json")
    else:
        resume_text = str(req.resume)

    if not resume_text.strip():
        raise HTTPException(status_code=422, detail="Resume text is empty after flattening")

    try:
        vecs = _encode_text(resume_text, boost=req.boost_weight)
    except Exception as e:
        log.error(f"Resume encoding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    log.info("Resume encoded successfully")
    return EncodeResumeResponse(tfidf=vecs["tfidf"], bert=vecs["bert"])


@app.post(
    "/vectorize-hnsw",
    tags=["pipeline"],
    summary="Process internships.mod-unvectorised → vectorise, HNSW graph, move to internships",
)
def vectorize_hnsw_endpoint():
    """
    Triggers the internship vectorization & HNSW graph maintenance pipeline.
    Reads from 'internships.mod-unvectorised', encodes, inserts into HNSW graph,
    persists in 'internships.graph', moves to 'internships', and cleans up.
    """
    try:
        from vectorizer_hnsw_pipeline import run_internship_vectorizer_hnsw_pipeline
        stats = run_internship_vectorizer_hnsw_pipeline()
        return {"status": "ok", "stats": stats}
    except Exception as e:
        log.error(f"HNSW Vectorizer Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
