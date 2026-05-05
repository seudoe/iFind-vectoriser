"""
task4_recommender.py
────────────────────
Hybrid Internship Recommender — Task 4

Formula:
    Final Score = (TF-IDF cosine × 0.4) + (BERT cosine × 0.6)

Depends on:
    task1_module.py
    task2_module.py
    task1-tfidf_vectorizer.pkl
    bert_model_saved/
    skill_clusters.json        (from task3 notebook output)
    skill_to_category.json     (from task3 notebook output)

Run:
    python task4_recommender.py
"""

import os
import sys
import json
import pickle
import textwrap
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(__file__))

from task1_module import flatten, tfidf_transform, load_vectorizer
from task2_module import load_bert_model, bert_transform
from task3_internship_profiles_module import (
    boost_tfidf_vector,
    generate_internship_profile,
    InternshipProfile,
    detect_category,
    extract_skills_from_text,
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION A — Load all models & clusters
# ─────────────────────────────────────────────────────────────────────────────

def load_all(
    tfidf_path: str = "task1-tfidf_vectorizer.pkl",
    bert_path:  str = "bert_model_saved",
    clusters_path: str = "skill_clusters.json",
    s2c_path:      str = "skill_to_category.json",
):
    print("Loading TF-IDF vectorizer …")
    tfidf_vec = load_vectorizer(tfidf_path)
    print(f"  vocab size : {len(tfidf_vec.vocabulary_):,}")

    print("Loading BERT model …")
    bert_mdl = load_bert_model(bert_path)

    # ── Auto-generate skill clusters if JSON files are missing ──
    if not os.path.exists(clusters_path) or not os.path.exists(s2c_path):
        print("\n⚠️  skill_clusters.json / skill_to_category.json not found.")
        print("   Generating them now from resume_data.csv …")
        _generate_skill_clusters(tfidf_vec, clusters_path, s2c_path)

    print("Loading skill clusters …")
    with open(clusters_path) as f:
        skill_clusters = json.load(f)
    with open(s2c_path) as f:
        skill_to_category = json.load(f)
    print(f"  {len(skill_clusters)} clusters, {len(skill_to_category)} skill→category mappings")

    return tfidf_vec, bert_mdl, skill_clusters, skill_to_category


def _generate_skill_clusters(tfidf_vec, clusters_path: str, s2c_path: str,
                              csv_path: str = "resume_data.csv", top_n: int = 40):
    """Generate skill_clusters.json and skill_to_category.json from resume_data.csv."""
    import pandas as pd
    from collections import defaultdict

    df = pd.read_csv(csv_path)
    feature_names = np.array(tfidf_vec.get_feature_names_out())
    skill_clusters = {}

    for category in df["Category"].unique():
        texts   = df[df["Category"] == category]["Resume"].fillna("").tolist()
        cleaned = [flatten(t, mode="text") for t in texts]
        matrix  = tfidf_vec.transform(cleaned)
        scores  = np.asarray(matrix.sum(axis=0)).flatten()
        top_idx = scores.argsort()[-top_n:][::-1]
        skill_clusters[category] = feature_names[top_idx].tolist()

    skill_to_category = {}
    for cat, skills in skill_clusters.items():
        for skill in skills:
            if skill not in skill_to_category:
                skill_to_category[skill] = cat

    with open(clusters_path, "w") as f:
        json.dump(skill_clusters, f, indent=2)
    with open(s2c_path, "w") as f:
        json.dump(skill_to_category, f, indent=2)

    print(f"  ✅ Saved {clusters_path} and {s2c_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION B — Resume vectoriser (mirrors internship pipeline)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResumeProfile:
    """Vectorised representation of a candidate resume."""
    name:             str
    raw_text:         str
    flat_text:        str          = field(repr=False)
    detected_domain:  str          = "Unknown"
    skills_found:     List[str]    = field(default_factory=list)
    tfidf_vector:     np.ndarray   = field(default=None, repr=False)
    bert_vector:      np.ndarray   = field(default=None, repr=False)


def build_resume_profile(
    name:              str,
    resume_input,                   # dict (JSON schema) | str (plain text)
    tfidf_vectorizer,
    bert_model,
    skill_clusters:    dict,
    skill_to_category: dict,
    boost_weight:      float = 0.15,
    mode:              str   = "auto",
) -> ResumeProfile:
    """
    Convert a resume (JSON dict or plain text) into a ResumeProfile.

    The same boost logic used for internships is applied here so that
    a resume mentioning 'Vue' also gets weight for 'React', 'Angular' etc.
    This makes matching symmetric.
    """
    flat_text = flatten(resume_input, mode=mode)

    detected_domain = detect_category(flat_text, skill_to_category, skill_clusters)
    skills_found    = extract_skills_from_text(flat_text, skill_to_category)

    base_tfidf    = tfidf_transform(resume_input, tfidf_vectorizer, mode=mode)
    boosted_tfidf = boost_tfidf_vector(
        base_vector       = base_tfidf,
        text              = flat_text,
        vectorizer        = tfidf_vectorizer,
        skill_clusters    = skill_clusters,
        skill_to_category = skill_to_category,
        boost_weight      = boost_weight,
    )
    bert_vec = bert_transform(resume_input, bert_model, mode=mode)

    return ResumeProfile(
        name            = name,
        raw_text        = str(resume_input),
        flat_text       = flat_text,
        detected_domain = detected_domain,
        skills_found    = skills_found,
        tfidf_vector    = boosted_tfidf,
        bert_vector     = bert_vec,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION C — Scoring
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_score(
    resume:     ResumeProfile,
    internship: InternshipProfile,
    w_tfidf:    float = 0.4,
    w_bert:     float = 0.6,
) -> dict:
    """
    Compute the hybrid similarity score between one resume and one internship.

    Returns
    -------
    dict with keys:
        tfidf_score  : float  — cosine similarity on TF-IDF vectors
        bert_score   : float  — cosine similarity on BERT vectors
        final_score  : float  — weighted combination
    """
    tfidf_sim = float(cosine_similarity(
        resume.tfidf_vector.reshape(1, -1),
        internship.tfidf_vector.reshape(1, -1)
    )[0][0])

    bert_sim = float(cosine_similarity(
        resume.bert_vector.reshape(1, -1),
        internship.bert_vector.reshape(1, -1)
    )[0][0])

    final = (w_tfidf * tfidf_sim) + (w_bert * bert_sim)

    return {
        "tfidf_score": round(tfidf_sim, 4),
        "bert_score":  round(bert_sim,  4),
        "final_score": round(final,     4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION D — Main recommender function
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RecommendationResult:
    internship_title:  str
    detected_domain:   str
    required_skills:   List[str]
    tfidf_score:       float
    bert_score:        float
    final_score:       float
    recommended:       bool
    threshold:         float
    matched_skills:    List[str]   # skills present in BOTH resume and internship
    missing_skills:    List[str]   # internship skills NOT in resume


def recommend_internship(
    resume_input,
    internship_data:   List[dict],
    tfidf_vectorizer,
    bert_model,
    skill_clusters:    dict,
    skill_to_category: dict,
    threshold:         float = 0.35,
    w_tfidf:           float = 0.4,
    w_bert:            float = 0.6,
    boost_weight:      float = 0.15,
    resume_name:       str   = "Candidate",
    resume_mode:       str   = "auto",
) -> List[RecommendationResult]:
    """
    Core recommender function.

    Parameters
    ----------
    resume_input     : dict | str  — the candidate's resume
    internship_data  : list of dicts, each with keys:
                         'title'       (str)
                         'description' (str)
    tfidf_vectorizer : fitted TfidfVectorizer
    bert_model       : loaded SentenceTransformer
    skill_clusters   : { category: [skills] }
    skill_to_category: { skill: category }
    threshold        : float — minimum final_score to be "Recommended"
                       (default 0.35 — lower than the PDF's 0.75 because
                        cosine similarity on real resumes rarely exceeds 0.5)
    w_tfidf          : weight for TF-IDF component (default 0.4)
    w_bert           : weight for BERT component   (default 0.6)
    boost_weight     : cluster sibling boost magnitude
    resume_name      : label for the candidate
    resume_mode      : 'json' | 'text' | 'auto'

    Returns
    -------
    List[RecommendationResult] sorted by final_score descending
    """
    # ── 1. Vectorise the resume ──
    print(f"\nVectorising resume for: {resume_name} …")
    resume_profile = build_resume_profile(
        name              = resume_name,
        resume_input      = resume_input,
        tfidf_vectorizer  = tfidf_vectorizer,
        bert_model        = bert_model,
        skill_clusters    = skill_clusters,
        skill_to_category = skill_to_category,
        boost_weight      = boost_weight,
        mode              = resume_mode,
    )
    print(f"  Detected domain : {resume_profile.detected_domain}")
    print(f"  Skills found    : {resume_profile.skills_found[:10]}")

    # ── 2. Vectorise each internship & score ──
    results = []
    print(f"\nScoring {len(internship_data)} internship(s) …")

    for item in internship_data:
        title       = item.get("title", "Untitled")
        description = item.get("description", "")

        internship_profile = generate_internship_profile(
            title             = title,
            description       = description,
            tfidf_vectorizer  = tfidf_vectorizer,
            bert_model        = bert_model,
            skill_clusters    = skill_clusters,
            skill_to_category = skill_to_category,
            boost_weight      = boost_weight,
        )

        scores = hybrid_score(resume_profile, internship_profile, w_tfidf, w_bert)

        resume_skill_set     = set(resume_profile.skills_found)
        internship_skill_set = set(internship_profile.required_skills)
        matched_skills = sorted(resume_skill_set & internship_skill_set)
        missing_skills = sorted(internship_skill_set - resume_skill_set)

        results.append(RecommendationResult(
            internship_title = title,
            detected_domain  = internship_profile.detected_category,
            required_skills  = internship_profile.required_skills,
            tfidf_score      = scores["tfidf_score"],
            bert_score       = scores["bert_score"],
            final_score      = scores["final_score"],
            recommended      = scores["final_score"] >= threshold,
            threshold        = threshold,
            matched_skills   = matched_skills,
            missing_skills   = missing_skills,
        ))

    # ── 3. Sort by final score descending ──
    results.sort(key=lambda r: r.final_score, reverse=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION E — Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def print_results(results: List[RecommendationResult], resume_name: str = "Candidate"):
    width = 70
    print("\n" + "═" * width)
    print(f"  RECOMMENDATION REPORT — {resume_name}")
    print("═" * width)

    recommended   = [r for r in results if r.recommended]
    not_recommend = [r for r in results if not r.recommended]

    print(f"\n  ✅  RECOMMENDED  ({len(recommended)} internship(s))")
    print("─" * width)
    for r in recommended:
        _print_one(r)

    print(f"\n  ❌  NOT RECOMMENDED  ({len(not_recommend)} internship(s))")
    print("─" * width)
    for r in not_recommend:
        _print_one(r)

    print("═" * width + "\n")


def _print_one(r: RecommendationResult):
    status = "✅ RECOMMENDED" if r.recommended else "❌ REJECTED"
    print(f"\n  [{status}]  {r.internship_title}")
    print(f"    Domain         : {r.detected_domain}")
    print(f"    Final Score    : {r.final_score:.4f}  "
          f"(TF-IDF={r.tfidf_score:.4f} × 0.4  +  BERT={r.bert_score:.4f} × 0.6)")
    print(f"    Threshold      : {r.threshold}")
    print(f"    Matched skills : {r.matched_skills or '—'}")
    print(f"    Missing skills : {r.missing_skills[:8] or '—'}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION F — Demo run
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RESUME = """
Experienced software developer with 2 years of internship experience.
Strong Python skills with hands-on work in Django, Flask, and REST API design.
Built data pipelines using pandas and scikit-learn for ML model evaluation.
Familiar with PostgreSQL, MongoDB, and Redis for data storage.
Deployed applications on AWS using Docker containers and CI/CD pipelines.
Comfortable with Git, Linux, and Agile/Scrum workflows.
Some exposure to React.js for frontend integration tasks.
"""

SAMPLE_INTERNSHIPS = [
    {
        "title": "Backend Engineering Intern",
        "description": """
            We are looking for a Backend Engineering Intern with strong Python skills.
            Experience with Django or Flask, REST APIs, and PostgreSQL is required.
            Familiarity with Docker containers and AWS deployment is a strong plus.
            You will design microservices, write unit tests, and participate in code reviews.
        """,
    },
    {
        "title": "Data Science Intern",
        "description": """
            Join our data team as a Data Science Intern. You should be comfortable with
            Python, pandas, and scikit-learn for data analysis and ML model building.
            Knowledge of deep learning frameworks like TensorFlow or PyTorch is a plus.
            You will work on feature engineering, model evaluation, and A/B testing.
        """,
    },
    {
        "title": "Frontend Engineering Intern",
        "description": """
            We need a creative Frontend Intern with hands-on experience in React.js.
            Strong understanding of JavaScript ES6+, HTML5, and CSS3 is essential.
            Experience with TypeScript, Redux state management, and responsive design
            is highly valued. Bonus: Figma for design collaboration.
        """,
    },
    {
        "title": "Cloud & DevOps Intern",
        "description": """
            Looking for a DevOps Intern to help manage our Kubernetes clusters on GCP.
            Must know Terraform for infrastructure-as-code and Jenkins for CI/CD.
            Experience with Docker, Helm charts, and monitoring tools (Prometheus, Grafana).
            Scripting in Bash or Python is required.
        """,
    },
    {
        "title": "iOS Mobile Development Intern",
        "description": """
            Join our mobile team as an iOS Intern. You must know Swift and UIKit.
            Experience with Xcode, CocoaPods, and App Store deployment is required.
            Familiarity with SwiftUI and Core Data is a strong plus.
            No backend experience needed — purely native iOS development.
        """,
    },
    {
        "title": "Cybersecurity Intern",
        "description": """
            We are hiring a Cybersecurity Intern to assist with penetration testing
            and vulnerability assessments. Knowledge of network security protocols,
            encryption standards, and OWASP Top 10 is required.
            Experience with Kali Linux, Burp Suite, or Metasploit is a plus.
        """,
    },
]


if __name__ == "__main__":
    # ── Load everything ──
    tfidf_vec, bert_mdl, skill_clusters, skill_to_category = load_all(
        tfidf_path    = "task1-tfidf_vectorizer.pkl",
        bert_path     = "bert_model_saved",
        clusters_path = "skill_clusters.json",
        s2c_path      = "skill_to_category.json",
    )

    # ── Run recommender ──
    results = recommend_internship(
        resume_input      = SAMPLE_RESUME,
        internship_data   = SAMPLE_INTERNSHIPS,
        tfidf_vectorizer  = tfidf_vec,
        bert_model        = bert_mdl,
        skill_clusters    = skill_clusters,
        skill_to_category = skill_to_category,
        threshold         = 0.35,       # tune this — see note below
        resume_name       = "Sample Candidate",
        resume_mode       = "text",
    )

    # ── Print results ──
    print_results(results, resume_name="Sample Candidate")

    # ── Raw scores table ──
    print("\nFull score table (sorted by final score):")
    print(f"{'Internship':<35} {'TF-IDF':>8} {'BERT':>8} {'Final':>8} {'Result'}")
    print("─" * 75)
    for r in results:
        status = "✅" if r.recommended else "❌"
        print(f"{r.internship_title:<35} {r.tfidf_score:>8.4f} {r.bert_score:>8.4f} "
              f"{r.final_score:>8.4f}  {status}")
