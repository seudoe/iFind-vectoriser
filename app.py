"""
app.py — iFind vectoriser: encoding, HNSW graph and student recommendations
──────────────────────────────────────────────────────────────────────────
  POST /encode-internships  — batch encode internship descriptions → vectors
  POST /encode-resume       — encode one resume → vectors; with user_id (+ recommend) it also saves them on the
                              student and runs the student through the whole graph
  POST /vectorize-hnsw      — scraped internships: vectorize approved staged docs, add to graph, move to
                              `internships`, then run every student through the graph
  POST /vectorize-platform  — employer-posted internships: vectorize in place, add to graph, then run every
                              student through the graph
  POST /recommend-all       — run every student (or user_ids) through the graph (admin refresh)
  POST /search-internships, /rebuild-graph

The recommendation code is recommender.py. Full-graph traversal per student happens only on those triggers.

Deploy: Hugging Face Spaces → Docker → port 7860
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path

from localrun import load_env

load_env(Path(__file__).resolve().parent)   # .env (+ .env.local for local runs) before any module reads os.environ

from services import allowed_origins

from task1_module import flatten, tfidf_transform, load_vectorizer
from task2_module import load_bert_model, bert_transform, bert_batch_transform
from task3_internship_profiles_module import boost_tfidf_vector

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

from ifind_log import RequestLogMiddleware, get_logger, setup

setup("vectoriser")
log = get_logger("app")

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
        log.info("TF-IDF loaded", extra={"f": {"vocab": len(MODELS["tfidf"].vocabulary_)}})

        MODELS["bert"] = load_bert_model(bert_path)
        log.info("BERT loaded")

        with open(clusters_path) as f:
            MODELS["clusters"] = json.load(f)
        with open(s2c_path) as f:
            MODELS["s2c"] = json.load(f)
        log.info("skill clusters loaded", extra={"f": {"domains": len(MODELS["clusters"])}})

        MODELS["ready"] = True
        log.ok("all models ready")

    except Exception as e:
        log.error("startup failed", extra={"f": {"error": str(e)}})
        MODELS["ready"] = False
        MODELS["error"] = str(e)

    yield
    MODELS.clear()


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title    = "iFind Vectoriser",
    version  = "2.0.0",
    lifespan = lifespan,
)

app.add_middleware(RequestLogMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = allowed_origins(),   # MAIN_PROJECT_URL + INTERNSCRAPER_URL (empty = no browser origin)
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
    user_id: Optional[str] = Field(
        default=None,
        description="Student _id. When set, the vectors are saved on the user (resume.tfidf_vector / bert_vector).",
    )
    recommend: bool = Field(
        default=False,
        description="With user_id: after vectorisation run the student through the WHOLE graph and save "
                    "recommendedInternships.",
    )


class EncodeResumeResponse(BaseModel):
    tfidf: List[float]   # 15,000 floats
    bert:  List[float]   # 768 floats
    recommended: Optional[int] = None   # how many internships were recommended (only when recommend ran)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
def root():
    return {
        "status":  "ready" if MODELS.get("ready") else "loading",
        "service": "iFind Vector Encoder",
        "endpoints": ["/encode-internships", "/encode-resume", "/vectorize-hnsw", "/vectorize-platform", "/recommend-all"],
    }


@app.get("/health", tags=["health"])
def health():
    return {
        "ready": MODELS.get("ready", False),
        "error": MODELS.get("error"),
    }


def _build_info() -> dict:
    """Deployed commit: GIT_SHA (set at build/deploy), else a GIT_SHA file next to this module, else git, else "dev"."""
    import subprocess
    from datetime import datetime, timezone
    sha = os.getenv("GIT_SHA", "").strip()
    if not sha:
        f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GIT_SHA")
        if os.path.exists(f):
            sha = open(f).read().strip()
    if not sha:
        try:
            sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=3,
                                 cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
        except Exception:
            sha = ""
    return {"service": "vectoriser", "commit": (sha or "dev")[:7], "builtAt": os.getenv("BUILD_TIME") or None}


_BUILD = _build_info()


@app.get("/version")
def version():
    return _BUILD


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

    log.info("internships encoded", extra={"f": {"count": len(results)}})
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

    log.ok("resume encoded")
    recommended = None
    if req.user_id:
        try:
            from bson import ObjectId
            from vectorizer_hnsw_pipeline import get_db
            db = get_db()
            db["users"].update_one(
                {"_id": ObjectId(req.user_id)},
                {"$set": {"resume.tfidf_vector": vecs["tfidf"], "resume.bert_vector": vecs["bert"]}},
            )
            if req.recommend:
                from recommender import recommend_user
                recommended = recommend_user(req.user_id, vecs["tfidf"], vecs["bert"], db)
        except Exception as e:
            log.error(f"Saving resume vectors / recommending failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    return EncodeResumeResponse(tfidf=vecs["tfidf"], bert=vecs["bert"], recommended=recommended)


class VectorizeHnswRequest(BaseModel):
    ids: Optional[List[str]] = Field(default=None, description="Restrict to these staging _ids")
    background: bool = Field(default=False, description="Return immediately and run server-side")
    recommend: bool = Field(default=True, description="After indexing, run every student through the graph (in the background)")


def _scraped_then_recommend(ids, recommend: bool):
    from recommender import run_fanout
    from vectorizer_hnsw_pipeline import run_internship_vectorizer_hnsw_pipeline
    stats = run_internship_vectorizer_hnsw_pipeline(ids=ids)
    if recommend and stats.get("processed"):
        run_fanout()
    return stats


class SearchRequest(BaseModel):
    vector: List[float]
    k: int = Field(default=20, ge=1, le=200)


@app.post(
    "/vectorize-hnsw",
    tags=["pipeline"],
    summary="Vectorise APPROVED internships.mod-unvectorised docs, insert into HNSW graph, move to internships",
)
def vectorize_hnsw_endpoint(req: VectorizeHnswRequest = VectorizeHnswRequest(), background_tasks: BackgroundTasks = None):
    """
    Processes only moderation.status in (auto_approved, manually_approved).
    Non-approved docs stay in staging untouched. Afterwards (recommend=true) every student is run through the graph.
    """
    from vectorizer_hnsw_pipeline import run_internship_vectorizer_hnsw_pipeline

    if req.background and background_tasks is not None:
        background_tasks.add_task(_scraped_then_recommend, req.ids, req.recommend)
        return {"status": "queued"}
    try:
        stats = run_internship_vectorizer_hnsw_pipeline(ids=req.ids)
        if req.recommend and stats.get("processed") and background_tasks is not None:
            from recommender import run_fanout
            background_tasks.add_task(run_fanout)
        return {"status": "ok", "stats": stats}
    except Exception as e:
        log.error(f"HNSW Vectorizer Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class VectorizePlatformRequest(BaseModel):
    ids: List[str] = Field(..., min_length=1, max_length=50, description="_ids in internships.this-platform")
    background: bool = Field(default=False, description="Return immediately and run server-side")
    recommend: bool = Field(default=True, description="After indexing, run every student through the graph (in the background)")


def _platform_then_recommend(ids, recommend: bool):
    from recommender import run_fanout
    from vectorizer_hnsw_pipeline import run_platform_vectorizer_pipeline
    stats = run_platform_vectorizer_pipeline(ids=ids)
    if recommend and stats.get("processed"):
        run_fanout()
    return stats


@app.post(
    "/vectorize-platform",
    tags=["pipeline"],
    summary="Vectorise APPROVED employer-posted internships in place (internships.this-platform)",
)
def vectorize_platform_endpoint(req: VectorizePlatformRequest, background_tasks: BackgroundTasks = None):
    """
    ifind-source internships are not staged or moved: vectors are written onto the document where it lives and
    its id is added to the HNSW graph. Only moderation.status in (auto_approved, manually_approved) is processed.
    Afterwards (recommend=true) every student is run through the graph.
    """
    from vectorizer_hnsw_pipeline import run_platform_vectorizer_pipeline

    if req.background and background_tasks is not None:
        background_tasks.add_task(_platform_then_recommend, req.ids, req.recommend)
        return {"status": "queued"}
    try:
        stats = run_platform_vectorizer_pipeline(ids=req.ids)
        if req.recommend and stats.get("processed") and background_tasks is not None:
            from recommender import run_fanout
            background_tasks.add_task(run_fanout)
        return {"status": "ok", "stats": stats}
    except Exception as e:
        log.error(f"Platform vectorizer failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class RecommendAllRequest(BaseModel):
    user_ids: Optional[List[str]] = Field(default=None, max_length=500, description="Only these students; omit for everyone")
    background: bool = Field(default=True, description="Return immediately and run server-side")


@app.post(
    "/recommend-all",
    tags=["pipeline"],
    summary="Run every student (or user_ids) through the whole HNSW graph and save their recommendations",
)
def recommend_all(req: RecommendAllRequest = RecommendAllRequest(), background_tasks: BackgroundTasks = None):
    from recommender import run_fanout

    if req.background and background_tasks is not None:
        background_tasks.add_task(run_fanout, req.user_ids)
        return {"status": "queued"}
    try:
        return {"status": "ok", "stats": run_fanout(req.user_ids)}
    except Exception as e:
        log.error(f"Recommend-all failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search-internships", tags=["pipeline"], summary="k-NN over the internship HNSW graph")
def search_internships(req: SearchRequest):
    try:
        from vectorizer_hnsw_pipeline import search_graph
        return {"results": [{"id": i, "score": sc} for i, sc in search_graph(req.vector, req.k)]}
    except Exception as e:
        log.error(f"Graph search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rebuild-graph", tags=["pipeline"], summary="Rebuild the HNSW graph from live internships' stored vectors")
def rebuild_graph():
    try:
        from vectorizer_hnsw_pipeline import HNSWIndexManager, PIPELINE_LOCK, get_db
        with PIPELINE_LOCK:
            db = get_db()
            n = HNSWIndexManager().rebuild_from_db(db)
        return {"status": "ok", "indexed": n}
    except Exception as e:
        log.error(f"Graph rebuild failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Local run: python app.py  (fixed port 6000; the Dockerfile starts uvicorn itself on 7860 and never gets here)
    import uvicorn
    from localrun import banner, pick_port

    port = pick_port("vectoriser", int(os.environ.get("PORT", 6000)), "ifind and internScraper (their VECTORISER_URL)")
    banner("vectoriser", port, {
        "Allowed origins (CORS)": ", ".join(allowed_origins()) or None,
        "MongoDB": "configured" if os.environ.get("MONGODB_URI") else "MONGODB_URI is NOT set",
        "Note": "first start downloads BERT; /health says ready:false until it is loaded",
    })
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False, log_config=None)
