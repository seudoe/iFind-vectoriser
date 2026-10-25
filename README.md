---
title: VectorisationResume
emoji: 🏃
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
short_description: Vectoriser, HNSW graph and student recommendations
---

# vectoriser

FastAPI service that turns internships and resumes into vectors, keeps the HNSW k-NN graph, and writes each student's recommendations.

- **Encoding:** a TF-IDF vector (boosted skills, `task1-tfidf_vectorizer.pkl`, `skill_clusters.json`, `skill_to_category.json`) plus a 768-d BERT vector (`all-mpnet-base-v2`, downloaded from the HF hub at startup).
- **Graph:** an HNSW index over internship BERT vectors. The serialized graph is stored in MongoDB (`internships.graph`), not on disk.
- **Recommendations:** `recommender.py` runs every student through the graph and saves the result on the user document.

Deployed as the Hugging Face Space `seudoe/vectorisationResume`: `https://seudoe-vectorisationresume.hf.space` (Docker, port 7860).

## Endpoints

Interactive docs at `/docs`. No auth header; the Space URL is the only gate, so keep it unlisted. CORS allows `MAIN_PROJECT_URL` and `INTERNSCRAPER_URL`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Status (`ready` / `loading`) and the endpoint list |
| `GET` | `/health` | `{"ready": bool, "error": str \| null}` |
| `POST` | `/encode-internships` | Batch-encode internship descriptions into TF-IDF + BERT vectors (accepts raw MongoDB internship docs) |
| `POST` | `/encode-resume` | Encode one student's resume into TF-IDF + BERT vectors |
| `POST` | `/vectorize-hnsw` | Vectorize approved docs from `internships.mod-unvectorised`, insert into the graph, move them to `internships`. Body: `ids?`, `background?`, `recommend?` (default true: then re-runs every student) |
| `POST` | `/vectorize-platform` | Same, in place, for approved employer-posted internships (`internships.this-platform`) |
| `POST` | `/recommend-all` | Run every student (or `user_ids`, max 500) through the graph and save recommendations. `background` defaults to true |
| `POST` | `/search-internships` | k-NN over the graph. Body: `{"vector": [...], "k": 20}` (k 1–200) |
| `POST` | `/rebuild-graph` | Rebuild the graph from the vectors stored on live internships |

```bash
curl https://seudoe-vectorisationresume.hf.space/health
curl -X POST .../recommend-all -H 'content-type: application/json' -d '{"background": true}'
```

## Who calls it

- **internScraper** `POST /vectorize-hnsw` after a scrape push, and `/vectorize-platform` after an auto-approved employer post.
- **ifind** `/encode-resume` on resume upload; `/vectorize-hnsw`, `/vectorize-platform`, `/recommend-all` and `/rebuild-graph` from `lib/internship-vectorizer.ts` (approvals, employer publishes and the admin routes).

## Run locally

```bash
python -m venv .venv && .venv\Scripts\activate        # Python 3.11
pip install -r requirements.txt
uvicorn app:app --port 7860
```

First start downloads the BERT model (about 440 MB). Startup is slow; `/health` reports `ready: false` until the models load.

## Environment

| Name | Purpose |
|---|---|
| `MONGODB_URI`, `MONGODB_DB_NAME` | database (default `ifind`) |
| `MAIN_PROJECT_URL` | ifind origin(s) for CORS, comma-separated |
| `INTERNSCRAPER_URL` | internScraper base URL (CORS) |
| `TFIDF_PATH`, `BERT_PATH`, `CLUSTERS_PATH`, `S2C_PATH` | model file overrides |
| `LOG_LEVEL`, `NO_COLOR`, `FORCE_COLOR` | log output |

## Files

`app.py` (routes), `vectorizer_hnsw_pipeline.py` (encode, graph, move), `recommender.py` (student fan-out), `services.py` (peer URLs), `ifind_log.py` (shared log format), `env-manager.py` (pulls env from the env-manager service), `HF_DEPLOY.md` (file list for the Space).

## Deploy

Push to the Space remote (`hf-seudoe`) and to GitHub (`origin`: `seudoe/iFind-vectoriser`). The Space's `.gitattributes` tracks `*.pkl`, `*.bin` and `*.db` with LFS. See `HF_DEPLOY.md`.
