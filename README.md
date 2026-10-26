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
uv sync                                               # Python 3.11 venv from pyproject.toml + uv.lock (Windows gets the prebuilt chroma-hnswlib, no C++ tools)
uv run python app.py                                  # http://localhost:6000
uv run pytest                                         # unit tests (the API tests below need a running instance)
```

`python app.py` prints what is running and where. If 6000 is taken it prints a red warning and uses the next free port (then
ifind and internScraper have to point at it). Local-only settings go in `.env.local` (overrides `.env`, never deployed). The
Dockerfile still starts uvicorn on 7860.

First start downloads the BERT model (about 440 MB); `/health` reports `ready: false` until it is loaded.

## Functions

| Module | Function | What it does |
|---|---|---|
| `task1_module.py` | `flatten(data, mode)` | Resume JSON / internship text to one cleaned string (`parse_list_field`, `clean_text` helpers) |
| | `tfidf_transform(data, vectorizer, mode)` | 15,000-d TF-IDF vector from the pickled vectorizer |
| `task2_module.py` | `load_bert_model`, `bert_transform`, `bert_encode`, `bert_batch_transform` | 768-d L2-normalised `all-mpnet-base-v2` embedding (text truncated to 400 words) |
| `task3_internship_profiles_module.py` | `boost_tfidf_vector` | Adds `boost_weight` (0.15) to every skill-cluster sibling of a skill found in the text, re-normalises |
| | `detect_category`, `extract_skills_from_text`, `generate_internship_profile` | Category vote, skill tokens, full profile |
| `app.py` | `_encode_text` | flatten, TF-IDF, boost, BERT: used by both encode endpoints |
| `vectorizer_hnsw_pipeline.py` | `HNSWIndexManager` | Graph in `internships.graph` (`initialize_or_load`, `add_node`, `search`, `persist_graph`, `rebuild_from_db`); falls back to a pure-Python index when hnswlib is missing |
| | `run_internship_vectorizer_hnsw_pipeline(ids)` | Approved staging docs: encode, add to graph, move to `internships` |
| | `run_platform_vectorizer_pipeline(ids)` | Approved employer posts: encode from `name`/`company`/`summary`, write vectors in place, add to graph |
| | `search_graph`, `get_search_manager` | k-NN over the persisted graph (reloads when `updatedAt` changes) |
| `recommender.py` | `score_user` | Graph candidates, score = 0.4 TF-IDF + 0.6 BERT, threshold 0.1, plus location boost (city 0.08, state 0.04, remote 0.03), top 20 |
| | `recommend_user`, `recommend_users`, `run_fanout` | One student / selected students / everyone; saved to `users.recommendedInternships` (`recommendedList`, `recommendedScores`) |

## Tests

```bash
pytest tests/test_location_boost.py                    # unit, no services
# API + pipeline, against a scratch database (never "ifind"):
MONGODB_DB_NAME=ifind_vectest PORT=6100 uv run python app.py
VECTORISER_TEST_URL=http://localhost:6100 pytest tests/test_api_integration.py -v
```

`test_api_integration.py` covers: `/health`, `/`, `/encode-internships` (clean and raw MongoDB docs, shapes, normalisation, determinism, batch limit 70, validation), `/encode-resume` (parsed JSON, plain text, empty, with `user_id` + `recommend`), `/vectorize-hnsw` (approved moved with vectors, unapproved stays, unknown ids skipped), `/vectorize-platform`, `/search-internships`, `/recommend-all`, `/rebuild-graph`, and that a Python resume ranks Python roles first. It seeds and drops its own data.

Employer posts must carry `name` and `summary` (the ifind schema); `status: "published"` is required for them to be recommended.

## Environment

| Name | Purpose |
|---|---|
| `MONGODB_URI`, `MONGODB_DB_NAME` | database (default `ifind`) |
| `MAIN_PROJECT_URL` | ifind origin(s) for CORS, comma-separated |
| `INTERNSCRAPER_URL` | internScraper base URL (CORS) |
| `TFIDF_PATH`, `BERT_PATH`, `CLUSTERS_PATH`, `S2C_PATH` | model file overrides |
| `LOG_LEVEL`, `NO_COLOR`, `FORCE_COLOR` | log output |

## Files

`app.py` (routes), `vectorizer_hnsw_pipeline.py` (encode, graph, move), `recommender.py` (student fan-out), `services.py` (peer URLs), `ifind_log.py` (shared log format), `localrun.py` (local env/port helpers, imported by app.py so it must be in the Dockerfile COPY list), `env-manager.py` (pulls env from the env-manager service), `HF_DEPLOY.md` (file list for the Space).

## Deploy

Push to the Space remote (`hf-seudoe`) and to GitHub (`origin`: `seudoe/iFind-vectoriser`). The Space's `.gitattributes` tracks `*.pkl`, `*.bin` and `*.db` with LFS. See `HF_DEPLOY.md`.
