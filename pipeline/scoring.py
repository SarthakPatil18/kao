"""
Stage 4 — Multi-Signal Match Scoring

Fuses three independent signals into a single confidence score via sigmoid:

  1. face_similarity:  Cosine similarity between InsightFace 512-d embeddings
  2. phash_similarity: Perceptual hash (pHash) Hamming distance, normalized
  3. text_relevance:   Sentence-transformer cosine similarity on captions

Fusion formula (sigmoid squash, NOT flat weighted average):
  raw  = w1 * face_sim + w2 * phash_sim + w3 * text_sim
  conf = 1 / (1 + exp(-k * (raw - midpoint)))

See README for weight rationale and threshold justification.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

try:
    import imagehash
except ImportError:
    imagehash = None

try:
    from sentence_transformers import SentenceTransformer, util as st_util
except ImportError:
    SentenceTransformer = None
    st_util = None


# ---------------------------------------------------------------------------
# Scoring weights & thresholds (document these in README)
# ---------------------------------------------------------------------------

# Signal weights — face-dominant, but multi-signal
W_FACE = 0.55          # Primary biometric signal
W_PHASH = 0.25         # Catches literal image reuse independent of face model
W_TEXT = 0.20           # Catches topically-inconsistent false positives

# Sigmoid parameters
SIGMOID_K = 10.0        # Steepness — sharper transition around midpoint
SIGMOID_MID = 0.50      # Center — raw scores above 0.5 map to > 0.5 confidence

# Tier thresholds
TIER_HIGH = 0.85
TIER_MEDIUM = 0.60


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MatchScore:
    """Complete scoring result for one candidate."""
    face_similarity: float       # 0.0 – 1.0
    phash_similarity: float      # 0.0 – 1.0
    text_similarity: float       # 0.0 – 1.0
    raw_score: float             # Weighted sum
    confidence: float            # Sigmoid-squashed 0.0 – 1.0
    tier: str                    # "HIGH" / "MEDIUM" / "LOW"


# ---------------------------------------------------------------------------
# Text embedding model (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_text_model = None


def _get_text_model():
    """Lazy-load the sentence-transformer model."""
    global _text_model
    if _text_model is None:
        if SentenceTransformer is None:
            return None
        _text_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _text_model


# ---------------------------------------------------------------------------
# Individual signal functions
# ---------------------------------------------------------------------------

def compute_face_similarity(
    source_emb: np.ndarray,
    candidate_emb: np.ndarray,
) -> float:
    """Cosine similarity between two face embeddings.

    Args:
        source_emb: Source 512-d embedding.
        candidate_emb: Candidate 512-d embedding.

    Returns:
        Similarity in [0.0, 1.0].
    """
    dot = np.dot(source_emb, candidate_emb)
    norm_s = np.linalg.norm(source_emb)
    norm_c = np.linalg.norm(candidate_emb)
    if norm_s < 1e-8 or norm_c < 1e-8:
        return 0.0
    sim = dot / (norm_s * norm_c)
    # Clamp to [0, 1] — negative cosine similarity means very different
    return float(max(0.0, min(1.0, sim)))


def compute_phash_similarity(
    source_img: Image.Image,
    candidate_img: Image.Image,
) -> float:
    """Perceptual hash similarity via pHash Hamming distance.

    Args:
        source_img: PIL Image of source face.
        candidate_img: PIL Image of candidate face.

    Returns:
        Similarity in [0.0, 1.0] (1.0 = identical hash).
    """
    if imagehash is None:
        return 0.5  # Neutral fallback if imagehash not installed

    hash_s = imagehash.phash(source_img)
    hash_c = imagehash.phash(candidate_img)
    hamming = hash_s - hash_c  # Hamming distance (0 = identical)
    # pHash is 64 bits, so max distance is 64
    similarity = 1.0 - (hamming / 64.0)
    return float(max(0.0, min(1.0, similarity)))


def compute_text_similarity(
    query_context: str,
    candidate_caption: str,
) -> float:
    """Semantic similarity between query context and candidate caption.

    Uses all-MiniLM-L6-v2 sentence-transformer.

    Args:
        query_context: Context string (e.g. "face search for online presence").
        candidate_caption: Caption/snippet from search result.

    Returns:
        Similarity in [0.0, 1.0], or 0.5 (neutral) if no caption or model unavailable.
    """
    if not candidate_caption or not candidate_caption.strip():
        return 0.5  # Neutral — don't penalize or reward missing text

    model = _get_text_model()
    if model is None:
        return 0.5  # Neutral fallback

    try:
        emb_q = model.encode(query_context, convert_to_tensor=True)
        emb_c = model.encode(candidate_caption, convert_to_tensor=True)
        sim = st_util.cos_sim(emb_q, emb_c).item()
        return float(max(0.0, min(1.0, sim)))
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def _sigmoid(x: float, k: float = SIGMOID_K, mid: float = SIGMOID_MID) -> float:
    """Sigmoid squash: 1 / (1 + exp(-k * (x - mid)))."""
    exponent = -k * (x - mid)
    # Clamp to avoid overflow
    exponent = max(-500.0, min(500.0, exponent))
    return 1.0 / (1.0 + math.exp(exponent))


def _assign_tier(confidence: float) -> str:
    """Map confidence score to a tier label."""
    if confidence >= TIER_HIGH:
        return "HIGH"
    elif confidence >= TIER_MEDIUM:
        return "MEDIUM"
    else:
        return "LOW"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_candidate(
    source_emb: np.ndarray,
    candidate_emb: np.ndarray,
    source_img: Image.Image,
    candidate_img: Image.Image,
    candidate_caption: str = "",
    query_context: str = "face search for online presence verification",
) -> MatchScore:
    """Compute multi-signal fused confidence score for a candidate.

    Args:
        source_emb: Source face embedding (512-d).
        candidate_emb: Candidate face embedding (512-d).
        source_img: PIL Image of source face crop.
        candidate_img: PIL Image of candidate face crop.
        candidate_caption: Text caption/snippet from search result.
        query_context: Context string for text similarity.

    Returns:
        MatchScore with all individual and fused scores.
    """
    face_sim = compute_face_similarity(source_emb, candidate_emb)
    phash_sim = compute_phash_similarity(source_img, candidate_img)
    text_sim = compute_text_similarity(query_context, candidate_caption)

    raw = W_FACE * face_sim + W_PHASH * phash_sim + W_TEXT * text_sim
    confidence = _sigmoid(raw)
    tier = _assign_tier(confidence)

    return MatchScore(
        face_similarity=round(face_sim, 4),
        phash_similarity=round(phash_sim, 4),
        text_similarity=round(text_sim, 4),
        raw_score=round(raw, 4),
        confidence=round(confidence, 4),
        tier=tier,
    )


def score_face_only(
    source_emb: np.ndarray,
    candidate_emb: np.ndarray,
) -> MatchScore:
    """Score using face similarity only (when no image/text available).

    Useful for quick screening before downloading candidate images.
    """
    face_sim = compute_face_similarity(source_emb, candidate_emb)
    # Use face_sim as raw score, apply sigmoid
    raw = face_sim
    confidence = _sigmoid(raw)
    tier = _assign_tier(confidence)

    return MatchScore(
        face_similarity=round(face_sim, 4),
        phash_similarity=0.0,
        text_similarity=0.5,
        raw_score=round(raw, 4),
        confidence=round(confidence, 4),
        tier=tier,
    )
