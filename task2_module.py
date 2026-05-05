
"""
task2_module.py
───────────────
BERT Semantic Encoder using all-mpnet-base-v2.
Depends on: task1_module.py

Usage:
    from task2_module import load_bert_model, bert_transform, bert_batch_transform
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer

# Import flatten from Task 1 (must be in same directory)
from task1_module import flatten


# ────────────────────────────────────────────────────
# SECTION A: Model Loading
# ────────────────────────────────────────────────────

def load_bert_model(model_path: str = 'bert_model_saved', device: str = None):
    """
    Load BERT model from a saved folder OR download from HuggingFace.

    Parameters
    ----------
    model_path : str
        Path to saved folder (from Task 2 output) OR
        HuggingFace model name ('sentence-transformers/all-mpnet-base-v2')
    device : str, optional
        'cuda' or 'cpu'. Auto-detects if None.

    Returns
    -------
    SentenceTransformer
    """
    import torch
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading BERT model from: {model_path}  (device={device})')
    model = SentenceTransformer(model_path, device=device)
    print(f'BERT loaded. Output dims: {model.get_sentence_embedding_dimension()}')
    return model


# ────────────────────────────────────────────────────
# SECTION B: Encoding Functions
# ────────────────────────────────────────────────────

def _truncate(text: str, max_words: int = 400) -> str:
    """Keep first 200 + last 200 words if text is too long."""
    words = text.split()
    if len(words) > max_words:
        return ' '.join(words[:200] + words[-200:])
    return text


def bert_encode(text: str, model: SentenceTransformer) -> np.ndarray:
    """
    Encode a pre-cleaned string → 768-dim L2-normalised vector.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return np.zeros(model.get_sentence_embedding_dimension())
    return model.encode(
        _truncate(text),
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True
    )


def bert_transform(input_data, model: SentenceTransformer, mode: str = 'auto') -> np.ndarray:
    """
    Full pipeline: raw input → flatten (Task1) → BERT encode → 768-dim vector.

    Parameters
    ----------
    input_data : dict | pd.Series | str
    model      : loaded SentenceTransformer
    mode       : 'json' | 'csv' | 'text' | 'auto'

    Returns
    -------
    np.ndarray : shape (768,)
    """
    flat = flatten(input_data, mode=mode)
    return bert_encode(flat, model=model)


def bert_batch_transform(text_list: list, model: SentenceTransformer,
                          batch_size: int = 32) -> np.ndarray:
    """
    Encode a list of already-flattened strings in batches.

    Returns
    -------
    np.ndarray : shape (N, 768)
    """
    truncated = [_truncate(t) for t in text_list]
    return model.encode(
        truncated,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True
    )
