# Deploy to Hugging Face Spaces

## Files to upload (everything else can be ignored)

```
app.py
task1_module.py
task2_module.py
task3_internship_profiles_module.py
task4_recommender.py
task1-tfidf_vectorizer.pkl
skill_clusters.json
skill_to_category.json
bert_model_saved/               ← entire folder (418 MB, use Git LFS)
requirements.txt
Dockerfile
```

## Steps


1. Go to https://huggingface.co/new-space
2. Name it (e.g. `ifind-recommender`)
3. SDK → **Docker**
4. Clone the Space repo locally:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/ifind-recommender
   cd ifind-recommender
   ```
5. Copy all files listed above into the cloned folder
6. The bert_model_saved folder is 418 MB — enable Git LFS first:
   ```bash
   git lfs install
   git lfs track "bert_model_saved/*.safetensors"
   git add .gitattributes
   ```
7. Commit and push:
   ```bash
   git add .
   git commit -m "initial deploy"
   git push
   ```
8. HF will build the Docker image and start the server (~5 min first time)

## Your API URL

```
https://YOUR_USERNAME-ifind-recommender.hf.space
```

## Endpoints

| Method | Path | What it does |
|--------|------|--------------|
| GET | /health | Models loaded? |
| POST | /encode-internships | Batch encode internship text -> TF-IDF + BERT vectors |
| POST | /encode-resume | Encode one resume. With `user_id` the vectors are saved on the student; with `recommend: true` the student is also run through the whole HNSW graph and `recommendedInternships` is saved |
| POST | /vectorize-hnsw | Scraped internships: vectorize approved staged docs, add to the graph, move to `internships`; then every student is run through the graph (`recommend`, default true) |
| POST | /vectorize-platform | Employer-posted internships (`internships.this-platform`): vectorize in place, add to the graph; then every student is run through the graph |
| POST | /recommend-all | Run every student (or `user_ids`) through the graph (admin refresh) |
| POST | /search-internships, /rebuild-graph | k-NN over the graph / rebuild it from stored vectors |
| GET | /docs | Swagger UI |

Environment: `MONGODB_URI` (required, the vectoriser reads and writes the shared database), optional `MONGODB_DB_NAME` (default `ifind`).

The recommendation code is `recommender.py` (moved here from `ifind/lib/recommendation`). Callers: internScraper
(auto-approved scraped and employer posts) and ifind (moderator approvals, resume/profile changes, admin refresh).
