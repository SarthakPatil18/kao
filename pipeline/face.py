"""
Stage 2 — Face Detection & Embedding

Uses InsightFace (buffalo_l model, ONNX runtime) for face detection and
512-dimensional embedding extraction.

PRIVACY: Embeddings are NEVER written to disk in raw form.  Only a SHA-256
hash of the embedding may be logged for internal deduplication / debugging.
"""

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "buffalo_l"           # InsightFace model pack
MIN_FACE_SIZE = 112                # Minimum face crop dimension (px)
MIN_QUALITY_SCORE = 30             # Below this → reject
LAPLACIAN_WEIGHT = 0.5             # Weight for sharpness in quality score
RESOLUTION_WEIGHT = 0.3            # Weight for resolution
FACE_RATIO_WEIGHT = 0.2            # Weight for face-to-canvas ratio


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FaceResult:
    """Result of face detection and embedding extraction."""
    face_crop: np.ndarray                      # BGR face crop
    embedding: np.ndarray                      # 512-d float32 vector
    embedding_hash: str                        # SHA-256 of embedding bytes
    bbox: Tuple[int, int, int, int]            # (x1, y1, x2, y2)
    quality_score: int                         # 0–100
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

_model_cache: Optional[FaceAnalysis] = None


def init_model(providers: Optional[List[str]] = None) -> "FaceAnalysis":
    """Load the InsightFace model (cached after first call).

    Args:
        providers: ONNX runtime providers.  Defaults to CPU.

    Returns:
        Initialized FaceAnalysis instance.
    """
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if FaceAnalysis is None:
        raise ImportError(
            "insightface is not installed.  Run: pip install insightface onnxruntime"
        )

    if providers is None:
        providers = ["CPUExecutionProvider"]

    app = FaceAnalysis(name=MODEL_NAME, providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))
    _model_cache = app
    return app


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

def compute_quality_score(face_crop: np.ndarray) -> int:
    """Compute a quality score (0–100) for a face crop.

    Combines:
      - Sharpness: Laplacian variance (higher = sharper)
      - Resolution: sqrt(width * height) normalized
      - Face-canvas ratio: face area / image area (closer to 1 = tighter crop)

    Returns:
        Integer quality score 0–100.
    """
    h, w = face_crop.shape[:2]

    # Sharpness via Laplacian variance
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize: typical range 0–2000, cap at 500
    sharpness = min(laplacian_var / 500.0, 1.0)

    # Resolution: sqrt(w*h) normalized against 512px baseline
    resolution = min(np.sqrt(w * h) / 512.0, 1.0)

    # Face-canvas ratio (always 1.0 for a tight crop; penalizes large padding)
    # For InsightFace aligned crops this is typically high
    face_ratio = min(w * h / (max(w, h) ** 2), 1.0)

    score = (
        LAPLACIAN_WEIGHT * sharpness
        + RESOLUTION_WEIGHT * resolution
        + FACE_RATIO_WEIGHT * face_ratio
    ) * 100

    return int(np.clip(score, 0, 100))


# ---------------------------------------------------------------------------
# Embedding hashing
# ---------------------------------------------------------------------------

def hash_embedding(embedding: np.ndarray) -> str:
    """Compute SHA-256 of the raw embedding bytes.

    This hash is used ONLY for internal deduplication / debugging.
    The raw embedding is never persisted or transmitted.
    """
    return hashlib.sha256(embedding.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_and_embed(
    image: np.ndarray,
    model: Optional["FaceAnalysis"] = None,
) -> Optional[FaceResult]:
    """Detect face(s) in an image and extract a 512-d embedding.

    If multiple faces are found, the largest bounding box is selected.

    Args:
        image: BGR numpy array.
        model: Pre-initialized FaceAnalysis (will auto-init if None).

    Returns:
        FaceResult on success, None if no face detected or quality too low.
    """
    if model is None:
        model = init_model()

    faces = model.get(image)
    warnings: List[str] = []

    if not faces:
        return None

    if len(faces) > 1:
        warnings.append(f"Multiple faces detected ({len(faces)}). Using largest.")

    # Pick the largest face by bounding-box area
    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    x1, y1, x2, y2 = [int(c) for c in best.bbox]
    h, w = image.shape[:2]
    # Clamp to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    face_crop = image[y1:y2, x1:x2].copy()

    # Check minimum size
    fh, fw = face_crop.shape[:2]
    if fh < MIN_FACE_SIZE or fw < MIN_FACE_SIZE:
        warnings.append(
            f"Face too small ({fw}×{fh}px, minimum {MIN_FACE_SIZE}×{MIN_FACE_SIZE})."
        )
        return None

    embedding = best.embedding  # 512-d float32
    emb_hash = hash_embedding(embedding)
    quality = compute_quality_score(face_crop)

    if quality < MIN_QUALITY_SCORE:
        warnings.append(f"Quality score {quality} below threshold {MIN_QUALITY_SCORE}.")
        return None

    return FaceResult(
        face_crop=face_crop,
        embedding=embedding,
        embedding_hash=emb_hash,
        bbox=(x1, y1, x2, y2),
        quality_score=quality,
        warnings=warnings,
    )


def encode_face_from_file(image_path: str) -> Optional[FaceResult]:
    """Convenience wrapper: load an image file and run detection + embedding."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    return detect_and_embed(image)
