FROM python:3.11-slim

WORKDIR /app

# hnswlib compiles a C++ extension
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY task1_module.py .
COPY task2_module.py .
COPY task3_internship_profiles_module.py .
COPY app.py .
COPY ifind_log.py .
COPY localrun.py .
COPY vectorizer_hnsw_pipeline.py .
COPY recommender.py .
COPY services.py .

# Copy model artifacts
COPY task1-tfidf_vectorizer.pkl .
COPY skill_clusters.json .
COPY skill_to_category.json .
# bert_model_saved is NOT copied — model is downloaded from HuggingFace on first startup

# HF Spaces expects port 7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
