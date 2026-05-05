"""'
task1_module.py
───────────────
Unified Data Flattener + TF-IDF utilities.
Built for: saugataroyarghya/resume-dataset (structured multi-column CSV)
Import in Tasks 2, 3, 4:
    from task1_module import flatten, tfidf_transform, load_vectorizer
"""

import re
import ast
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


# ────────────────────────────────────────────
# A: Parsers
# ────────────────────────────────────────────

def parse_list_field(value) -> str:
    """Convert stringified Python list (or nested list) to plain string."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ''
    if isinstance(value, list):
        flat = []
        for item in value:
            if isinstance(item, list):
                flat.extend([str(x) for x in item if x is not None])
            elif item is not None:
                flat.append(str(item))
        return ' '.join(flat)
    text = str(value).strip()
    if text.startswith('['):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                flat = []
                for item in parsed:
                    if isinstance(item, list):
                        flat.extend([str(x) for x in item if x is not None])
                    elif item is not None:
                        flat.append(str(item))
                return ' '.join(flat)
        except Exception:
            text = re.sub(r"[\[\]'\"]", ' ', text)
    return text


def clean_text(text: str) -> str:
    """Strip markdown links, URLs, HTML, special chars. Lowercase."""
    if not isinstance(text, str):
        return ''
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s+#.]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


# ────────────────────────────────────────────
# B: Flatteners (edit _flatten_json to swap schema)
# ────────────────────────────────────────────

def _flatten_csv_row(row: pd.Series) -> str:
    parts = []
    obj = row.get('career_objective', '')
    if isinstance(obj, str) and obj.strip():
        parts.append(obj)
    skills_raw = parse_list_field(row.get('skills', ''))
    if skills_raw:
        parts.append(skills_raw)
        parts.append(skills_raw)
    for col in ['positions', 'responsibilities', 'related_skils_in_job',
                'degree_names', 'major_field_of_studies',
                'educational_institution_name', 'certification_skills',
                'extra_curricular_activity_types', 'languages']:
        val = parse_list_field(row.get(col, ''))
        if val:
            parts.append(val)
    return ' '.join(parts)


def _flatten_json(resume_dict: dict) -> str:
    """EDIT THIS if your JSON schema changes."""
    parts = []
    summary = resume_dict.get('Summary', '') or resume_dict.get('Objective', '')
    parts.append(str(summary))
    skills = resume_dict.get('Skills', [])
    skill_text = ' '.join([str(s) for s in skills if s]) if isinstance(skills, list) else str(skills)
    parts.append(skill_text)
    parts.append(skill_text)
    for job in resume_dict.get('Work History', []):
        if isinstance(job, dict):
            parts.append(f"{job.get('title','')  } {job.get('company','')} {job.get('description','')}")
        else:
            parts.append(str(job))
    for proj in resume_dict.get('Projects', []):
        if isinstance(proj, dict):
            parts.append(f"{proj.get('name','')} {proj.get('description','')}")
        else:
            parts.append(str(proj))
    parts.append(str(resume_dict.get('Education', '')))
    return ' '.join(parts)


def flatten(input_data, mode: str = 'auto') -> str:
    """Universal flattener. mode: 'csv'|'json'|'text'|'auto'."""
    if mode == 'auto':
        if isinstance(input_data, dict):        mode = 'json'
        elif isinstance(input_data, pd.Series): mode = 'csv'
        else:                                   mode = 'text'
    if mode == 'json':
        raw = _flatten_json(input_data)
    elif mode == 'csv':
        raw = _flatten_csv_row(input_data)
    else:
        raw = str(input_data)
    return clean_text(raw)


# ────────────────────────────────────────────
# C: Vectorizer I/O
# ────────────────────────────────────────────

def load_vectorizer(path: str = 'tfidf_vectorizer.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)


def tfidf_transform(input_data, vectorizer, mode='auto') -> np.ndarray:
    """Flatten → TF-IDF → L2-normalised dense vector. Shape: (vocab_size,)"""
    flat = flatten(input_data, mode=mode)
    return normalize(vectorizer.transform([flat]), norm='l2').toarray()[0]
