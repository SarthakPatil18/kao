"""
Tests for multi-signal scoring.

Tests cover:
  - Individual signal correctness (face, phash, text)
  - Sigmoid monotonicity and bounds
  - Tier assignment at boundaries
  - Known match/non-match scenarios for threshold justification
  - Fusion formula correctness
"""

import math
import numpy as np
import pytest
from pipeline.scoring import (
    compute_face_similarity,
    compute_phash_similarity,
    _sigmoid,
    _assign_tier,
    score_candidate,
    score_face_only,
    W_FACE,
    W_PHASH,
    W_TEXT,
    SIGMOID_K,
    SIGMOID_MID,
    TIER_HIGH,
    TIER_MEDIUM,
    MatchScore,
)


# ---------------------------------------------------------------------------
# Face similarity tests
# ---------------------------------------------------------------------------

class TestFaceSimilarity:
    """Cosine similarity between face embeddings."""

    def test_identical_embeddings(self):
        """Identical vectors → similarity = 1.0."""
        emb = np.random.randn(512).astype(np.float32)
        assert compute_face_similarity(emb, emb) == pytest.approx(1.0, abs=1e-4)

    def test_orthogonal_embeddings(self):
        """Orthogonal vectors → similarity ≈ 0.0."""
        e1 = np.zeros(512, dtype=np.float32)
        e2 = np.zeros(512, dtype=np.float32)
        e1[0] = 1.0
        e2[1] = 1.0
        assert compute_face_similarity(e1, e2) == pytest.approx(0.0, abs=1e-4)

    def test_opposite_embeddings_clamped(self):
        """Opposite vectors → clamped to 0.0 (not negative)."""
        emb = np.random.randn(512).astype(np.float32)
        assert compute_face_similarity(emb, -emb) == 0.0

    def test_zero_vector(self):
        """Zero vector → similarity = 0.0 (avoid division by zero)."""
        emb = np.random.randn(512).astype(np.float32)
        zero = np.zeros(512, dtype=np.float32)
        assert compute_face_similarity(emb, zero) == 0.0

    def test_similar_embeddings(self):
        """Similar vectors → high similarity."""
        emb = np.random.randn(512).astype(np.float32)
        noisy = emb + np.random.randn(512).astype(np.float32) * 0.1
        sim = compute_face_similarity(emb, noisy)
        assert sim > 0.8


# ---------------------------------------------------------------------------
# Sigmoid tests
# ---------------------------------------------------------------------------

class TestSigmoid:
    """Sigmoid squash function properties."""

    def test_midpoint_maps_to_half(self):
        """Input at midpoint → output = 0.5."""
        assert _sigmoid(SIGMOID_MID) == pytest.approx(0.5, abs=1e-4)

    def test_monotonically_increasing(self):
        """Higher input → higher output."""
        vals = [_sigmoid(x / 10.0) for x in range(0, 11)]
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1]

    def test_bounded_zero_one(self):
        """Output always in [0, 1]."""
        for x in np.linspace(-5, 5, 100):
            s = _sigmoid(float(x))
            assert 0.0 <= s <= 1.0

    def test_extreme_high(self):
        """Very high input → output ≈ 1.0."""
        assert _sigmoid(1.0) > 0.99

    def test_extreme_low(self):
        """Very low input → output ≈ 0.0."""
        assert _sigmoid(0.0) < 0.01


# ---------------------------------------------------------------------------
# Tier assignment tests
# ---------------------------------------------------------------------------

class TestTierAssignment:
    """Confidence tier boundary tests."""

    def test_high_tier(self):
        assert _assign_tier(0.85) == "HIGH"
        assert _assign_tier(0.99) == "HIGH"
        assert _assign_tier(1.0) == "HIGH"

    def test_medium_tier(self):
        assert _assign_tier(0.60) == "MEDIUM"
        assert _assign_tier(0.75) == "MEDIUM"
        assert _assign_tier(0.84) == "MEDIUM"

    def test_low_tier(self):
        assert _assign_tier(0.0) == "LOW"
        assert _assign_tier(0.30) == "LOW"
        assert _assign_tier(0.59) == "LOW"


# ---------------------------------------------------------------------------
# Fusion formula tests
# ---------------------------------------------------------------------------

class TestFusion:
    """Multi-signal fusion correctness."""

    def test_weights_sum_to_one(self):
        """Signal weights must sum to 1.0."""
        assert W_FACE + W_PHASH + W_TEXT == pytest.approx(1.0)

    def test_all_perfect_signals_high_confidence(self):
        """All signals at 1.0 → HIGH confidence."""
        raw = W_FACE * 1.0 + W_PHASH * 1.0 + W_TEXT * 1.0
        confidence = _sigmoid(raw)
        assert confidence >= TIER_HIGH
        assert _assign_tier(confidence) == "HIGH"

    def test_all_zero_signals_low_confidence(self):
        """All signals at 0.0 → LOW confidence."""
        raw = W_FACE * 0.0 + W_PHASH * 0.0 + W_TEXT * 0.0
        confidence = _sigmoid(raw)
        assert confidence < TIER_MEDIUM
        assert _assign_tier(confidence) == "LOW"

    def test_face_only_high(self):
        """High face sim + neutral others → reasonable confidence."""
        raw = W_FACE * 0.95 + W_PHASH * 0.5 + W_TEXT * 0.5
        confidence = _sigmoid(raw)
        assert confidence > 0.6  # Should be at least MEDIUM


# ---------------------------------------------------------------------------
# Known scenario evaluation (threshold justification)
# ---------------------------------------------------------------------------

class TestKnownScenarios:
    """Evaluate scoring on known match/non-match scenarios.

    These scenarios justify the chosen weights and thresholds.
    Results should be reported in the README.
    """

    def test_strong_match_scenario(self):
        """Same person, same photo reused: all signals high."""
        raw = W_FACE * 0.95 + W_PHASH * 0.90 + W_TEXT * 0.80
        confidence = _sigmoid(raw)
        assert _assign_tier(confidence) == "HIGH"
        assert confidence >= 0.85

    def test_same_person_different_photo(self):
        """Same person, different photo: high face sim, low phash."""
        raw = W_FACE * 0.85 + W_PHASH * 0.30 + W_TEXT * 0.70
        confidence = _sigmoid(raw)
        # Should still be MEDIUM or HIGH depending on face match quality
        assert _assign_tier(confidence) in ("MEDIUM", "HIGH")

    def test_different_person_same_context(self):
        """Different person in similar context: low face, moderate text."""
        raw = W_FACE * 0.20 + W_PHASH * 0.15 + W_TEXT * 0.75
        confidence = _sigmoid(raw)
        assert _assign_tier(confidence) == "LOW"

    def test_image_reuse_different_person(self):
        """Photo reuse (meme/stock photo): high phash, low face."""
        raw = W_FACE * 0.15 + W_PHASH * 0.95 + W_TEXT * 0.10
        confidence = _sigmoid(raw)
        # phash alone shouldn't push to HIGH — face weight dominates
        assert _assign_tier(confidence) == "LOW"

    def test_no_signals_at_all(self):
        """Complete non-match: everything low."""
        raw = W_FACE * 0.05 + W_PHASH * 0.05 + W_TEXT * 0.10
        confidence = _sigmoid(raw)
        assert _assign_tier(confidence) == "LOW"
        assert confidence < 0.1


class TestScoreFaceOnly:
    """Score face only (quick screening)."""

    def test_high_face_gives_high_score(self):
        emb = np.random.randn(512).astype(np.float32)
        result = score_face_only(emb, emb)
        assert result.tier == "HIGH"
        assert result.face_similarity > 0.99

    def test_low_face_gives_low_score(self):
        e1 = np.random.randn(512).astype(np.float32)
        e2 = np.random.randn(512).astype(np.float32)
        result = score_face_only(e1, e2)
        # Random vectors should have low similarity
        assert result.face_similarity < 0.5
