"""
task3_internship_profiles_module.py
────────────────────────────────────
Extracted from task3_internship_profiles.ipynb for import in Task 4.
Contains: boost engine, InternshipProfile, generate_internship_profile.

Usage:
    from task3_internship_profiles_module import (
        boost_tfidf_vector,
        generate_internship_profile,
        InternshipProfile,
        detect_category,
        extract_skills_from_text,
        load_skill_clusters,
    )
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import List
from collections import defaultdict

from task1_module import flatten, tfidf_transform
from task2_module import bert_transform


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_skill_clusters(path: str = "skill_clusters.json") -> dict:
    with open(path) as f:
        return json.load(f)


def load_skill_to_category(path: str = "skill_to_category.json") -> dict:
    with open(path) as f:
        return json.load(f)


def build_skill_to_category_map(skill_clusters: dict) -> dict:
    """Invert cluster dict: { skill: category }. First occurrence wins."""
    skill_to_category = {}
    for category, skills in skill_clusters.items():
        for skill in skills:
            if skill not in skill_to_category:
                skill_to_category[skill] = category
    return skill_to_category


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def detect_category(flat_text: str,
                    skill_to_category: dict,
                    skill_clusters: dict) -> str:
    """Vote-based: category with most skill-token matches wins."""
    votes = defaultdict(int)
    tokens = set(flat_text.lower().split())
    for token in tokens:
        if token in skill_to_category:
            votes[skill_to_category[token]] += 1
    return max(votes, key=votes.get) if votes else "Unknown"


def extract_skills_from_text(flat_text: str,
                              skill_to_category: dict) -> List[str]:
    """Return all cluster-skill tokens found in the text."""
    tokens = set(flat_text.lower().split())
    return sorted([t for t in tokens if t in skill_to_category])


# ─────────────────────────────────────────────────────────────────────────────
# Boost engine
# ─────────────────────────────────────────────────────────────────────────────

def boost_tfidf_vector(base_vector: np.ndarray,
                       text: str,
                       vectorizer,
                       skill_clusters: dict,
                       skill_to_category: dict,
                       boost_weight: float = 0.15) -> np.ndarray:
    """
    Add cluster-sibling skill weights to a TF-IDF vector, then re-normalise.

    If the text mentions 'react', all Frontend siblings (vue, angular, js …)
    get +boost_weight in the vector — even if they weren't in the text.
    """
    vocab = vectorizer.vocabulary_
    boosted = base_vector.copy()
    tokens_in_text = set(text.lower().split())
    boosted_categories = set()

    for token in tokens_in_text:
        if token not in skill_to_category:
            continue
        category = skill_to_category[token]
        if category in boosted_categories:
            continue
        boosted_categories.add(category)
        for sibling in skill_clusters.get(category, []):
            if sibling in vocab and sibling not in tokens_in_text:
                boosted[vocab[sibling]] += boost_weight

    norm = np.linalg.norm(boosted)
    return boosted / norm if norm > 0 else boosted


# ─────────────────────────────────────────────────────────────────────────────
# InternshipProfile dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InternshipProfile:
    title:             str
    raw_description:   str
    flat_text:         str           = field(repr=False)
    detected_category: str           = "Unknown"
    required_skills:   List[str]     = field(default_factory=list)
    tfidf_vector:      np.ndarray    = field(default=None, repr=False)
    bert_vector:       np.ndarray    = field(default=None, repr=False)
    hybrid_vector:     np.ndarray    = field(default=None, repr=False)


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_internship_profile(
        title: str,
        description: str,
        tfidf_vectorizer,
        bert_model,
        skill_clusters: dict,
        skill_to_category: dict,
        boost_weight: float = 0.15,
) -> InternshipProfile:
    """
    Raw internship description → InternshipProfile with hybrid vectors.
    """
    flat_text         = flatten(description, mode="text")
    detected_category = detect_category(flat_text, skill_to_category, skill_clusters)
    required_skills   = extract_skills_from_text(flat_text, skill_to_category)

    base_tfidf    = tfidf_transform(description, tfidf_vectorizer, mode="text")
    boosted_tfidf = boost_tfidf_vector(
        base_vector       = base_tfidf,
        text              = flat_text,
        vectorizer        = tfidf_vectorizer,
        skill_clusters    = skill_clusters,
        skill_to_category = skill_to_category,
        boost_weight      = boost_weight,
    )
    bert_vec = bert_transform(description, bert_model, mode="text")

    return InternshipProfile(
        title             = title,
        raw_description   = description,
        flat_text         = flat_text,
        detected_category = detected_category,
        required_skills   = required_skills,
        tfidf_vector      = boosted_tfidf,
        bert_vector       = bert_vec,
        hybrid_vector     = np.concatenate([boosted_tfidf, bert_vec]),
    )
