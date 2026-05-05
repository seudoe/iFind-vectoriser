FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY task1_module.py .
COPY task2_module.py .
COPY task3_internship_profiles_module.py .
COPY task4_recommender.py .
COPY app.py .

# Copy model artifacts
COPY task1-tfidf_vectorizer.pkl .
COPY skill_clusters.json .
COPY skill_to_category.json .
COPY bert_model_saved/ ./bert_model_saved/

# HF Spaces expects port 7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
