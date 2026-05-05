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
    """EDIT THIS if your JSON schema changes.
    Handles both camelCase (iFind schema) and PascalCase (legacy) keys.
    """
    parts = []

    # Summary
    summary = (resume_dict.get('summary') or resume_dict.get('Summary')
               or resume_dict.get('Objective') or '')
    if summary:
        parts.append(str(summary))

    # Skills — iFind schema: list of {field, tools:[{name}]}
    skills = resume_dict.get('skills') or resume_dict.get('Skills') or []
    if isinstance(skills, list):
        skill_tokens = []
        for s in skills:
            if isinstance(s, dict):
                # iFind schema: {field, tools:[{name}]}
                if 'field' in s:
                    skill_tokens.append(str(s['field']))
                for tool in s.get('tools', []):
                    if isinstance(tool, dict) and tool.get('name'):
                        skill_tokens.append(str(tool['name']))
                # legacy schema: plain string list
            elif isinstance(s, str):
                skill_tokens.append(s)
        skill_text = ' '.join(skill_tokens)
        parts.append(skill_text)
        parts.append(skill_text)  # double-weight skills

    # Work history — iFind schema: workHistory
    for job in (resume_dict.get('workHistory') or resume_dict.get('Work History') or []):
        if isinstance(job, dict):
            parts.append(job.get('title', ''))
            parts.append(job.get('company', ''))
            parts.append(' '.join(job.get('responsibilities', [])))
            parts.append(' '.join(job.get('achievements', [])))
        else:
            parts.append(str(job))

    # Projects
    for proj in (resume_dict.get('projects') or resume_dict.get('Projects') or []):
        if isinstance(proj, dict):
            parts.append(proj.get('title', ''))
            parts.append(proj.get('role', ''))
            parts.append(' '.join(proj.get('techStack', [])))
            parts.append(' '.join(proj.get('description', [])))
            parts.append(proj.get('problemStatement') or '')
        else:
            parts.append(str(proj))

    # Education — iFind schema: {institution, field:{type, course}}
    for edu in (resume_dict.get('education') or resume_dict.get('Education') or []):
        if isinstance(edu, dict):
            parts.append(edu.get('institution', ''))
            field = edu.get('field', {})
            if isinstance(field, dict):
                parts.append(field.get('course', ''))
                parts.append(field.get('type', ''))
            elif isinstance(field, str):
                parts.append(field)
        else:
            parts.append(str(edu))

    # Certifications
    for cert in (resume_dict.get('certifications') or []):
        if isinstance(cert, dict):
            parts.append(cert.get('name', ''))
            parts.append(' '.join(cert.get('skillsEarned', [])))

    # Affiliations
    for aff in (resume_dict.get('affiliations') or []):
        if isinstance(aff, dict):
            parts.append(aff.get('organization', ''))
            parts.append(aff.get('role', ''))
            parts.append(' '.join(aff.get('impact', [])))

    # Awards
    for award in (resume_dict.get('awards') or []):
        if isinstance(award, dict):
            parts.append(award.get('name', ''))
            parts.append(award.get('justification', ''))

    # Interests
    for interest in (resume_dict.get('interests') or []):
        if isinstance(interest, dict):
            parts.append(interest.get('activity', ''))
            parts.append(interest.get('description', ''))

    return ' '.join(p for p in parts if p)


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
