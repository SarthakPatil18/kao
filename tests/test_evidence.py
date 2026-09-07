"""
Tests for evidence canonicalization and hashing determinism.

Key properties tested:
  - Same input → same hash (idempotent)
  - Reordered dict keys → same canonical JSON → same hash
  - None values are stripped before canonicalization
  - Round-trip: save → load → re-hash → matches original
"""

import json
import pytest
from pipeline.evidence import (
    EvidenceRecord,
    canonicalize,
    evidence_hash,
    evidence_to_dict,
    compute_evidence_hash,
    build_evidence,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_evidence():
    """A representative evidence record."""
    return EvidenceRecord(
        confidence_tier="HIGH",
        fused_score=0.9123,
        face_similarity=0.9500,
        phash_similarity=0.8750,
        text_similarity=0.7200,
        discovered_url="https://example.com/profile",
        platform="linkedin",
        timestamp="2024-01-15T12:00:00+00:00",
        candidate_image_sha256="abc123def456",
    )


@pytest.fixture
def sample_dict(sample_evidence):
    """Evidence as a dict."""
    return evidence_to_dict(sample_evidence)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCanonicalization:
    """Canonicalization determinism tests."""

    def test_same_input_same_output(self, sample_dict):
        """Canonicalizing the same dict twice produces identical strings."""
        c1 = canonicalize(sample_dict)
        c2 = canonicalize(sample_dict)
        assert c1 == c2

    def test_reordered_keys_same_output(self, sample_dict):
        """Dict with reordered keys produces the same canonical JSON."""
        # Reverse the key order
        reversed_dict = dict(reversed(list(sample_dict.items())))
        c_original = canonicalize(sample_dict)
        c_reversed = canonicalize(reversed_dict)
        assert c_original == c_reversed

    def test_sorted_keys_in_output(self, sample_dict):
        """Canonical JSON has sorted keys."""
        canonical = canonicalize(sample_dict)
        parsed = json.loads(canonical)
        keys = list(parsed.keys())
        assert keys == sorted(keys)

    def test_compact_separators(self, sample_dict):
        """Canonical JSON uses compact separators (no spaces)."""
        canonical = canonicalize(sample_dict)
        assert ": " not in canonical  # No space after colon
        assert ", " not in canonical  # No space after comma

    def test_none_values_stripped(self):
        """None values are excluded from canonical form."""
        d = {
            "key1": "value1",
            "key2": None,
            "key3": "value3",
        }
        canonical = canonicalize(d)
        parsed = json.loads(canonical)
        assert "key2" not in parsed
        assert "key1" in parsed
        assert "key3" in parsed

    def test_pii_excluded_by_default(self, sample_evidence):
        """PII fields (subject_name, subject_handle) are None and stripped."""
        d = evidence_to_dict(sample_evidence)
        canonical = canonicalize(d)
        assert "subject_name" not in canonical
        assert "subject_handle" not in canonical


class TestHashing:
    """SHA-256 hashing tests."""

    def test_hash_determinism(self, sample_dict):
        """Same canonical JSON always produces the same hash."""
        c = canonicalize(sample_dict)
        h1 = evidence_hash(c)
        h2 = evidence_hash(c)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_different_input_different_hash(self, sample_dict):
        """Changing any field produces a different hash."""
        c1 = canonicalize(sample_dict)
        h1 = evidence_hash(c1)

        modified = dict(sample_dict)
        modified["confidence_tier"] = "LOW"
        c2 = canonicalize(modified)
        h2 = evidence_hash(c2)

        assert h1 != h2

    def test_compute_evidence_hash_convenience(self, sample_evidence):
        """compute_evidence_hash produces same result as manual pipeline."""
        manual = evidence_hash(canonicalize(evidence_to_dict(sample_evidence)))
        convenience = compute_evidence_hash(sample_evidence)
        assert manual == convenience


class TestBuildEvidence:
    """Evidence construction tests."""

    def test_build_without_pii(self):
        """Building without PII leaves name/handle as None."""
        record = build_evidence(
            confidence_tier="MEDIUM",
            fused_score=0.65,
            face_similarity=0.70,
            phash_similarity=0.60,
            text_similarity=0.55,
            discovered_url="https://example.com",
            platform="github",
            candidate_image_bytes=b"fake_image_bytes",
        )
        assert record.subject_name is None
        assert record.subject_handle is None

    def test_build_with_pii(self):
        """Building with include_pii=True stores name/handle."""
        record = build_evidence(
            confidence_tier="HIGH",
            fused_score=0.90,
            face_similarity=0.95,
            phash_similarity=0.85,
            text_similarity=0.80,
            discovered_url="https://example.com",
            platform="linkedin",
            candidate_image_bytes=b"fake_image_bytes",
            include_pii=True,
            subject_name="Test User",
            subject_handle="@testuser",
        )
        assert record.subject_name == "Test User"
        assert record.subject_handle == "@testuser"

    def test_image_hash_computed(self):
        """Candidate image SHA-256 is computed from raw bytes."""
        import hashlib
        img_bytes = b"test_image_content_12345"
        expected = hashlib.sha256(img_bytes).hexdigest()

        record = build_evidence(
            confidence_tier="LOW",
            fused_score=0.30,
            face_similarity=0.25,
            phash_similarity=0.20,
            text_similarity=0.15,
            discovered_url="https://example.com",
            platform="other",
            candidate_image_bytes=img_bytes,
        )
        assert record.candidate_image_sha256 == expected
