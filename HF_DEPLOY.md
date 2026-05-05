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

| Method | Path        | What it does                              |
|--------|-------------|-------------------------------------------|
| GET    | /health     | Check if models are loaded                |
| POST   | /recommend  | Score resume against N internships        |
| POST   | /score      | Score resume against 1 internship         |
| GET    | /docs       | Auto-generated Swagger UI                 |

## Example call from Next.js

```typescript
const res = await fetch("https://YOUR_USERNAME-ifind-recommender.hf.space/recommend", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    resume: parsedResumeData,   // the parsedData object from your MongoDB user doc
    internships: [
      { title: "Backend Intern", description: "Python, Django, REST APIs..." },
      { title: "Data Science Intern", description: "pandas, scikit-learn, ML..." },
    ],
    threshold: 0.35,
    resume_name: "John Doe",
  }),
});

const data = await res.json();
// data.results → sorted by final_score, each has recommended: true/false
```

## Resume input format

You can send either:

**Plain text** (simplest):
```json
{ "resume": "Python developer with 2 years experience in Django..." }
```

**Parsed JSON** (from your resume parser — uses all fields):
```json
{
  "resume": {
    "summary": "...",
    "skills": [...],
    "workHistory": [...],
    "projects": [...]
  }
}
```
