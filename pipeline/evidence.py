"""
Stage 5 — Evidence Record Construction & Canonicalization

Builds a structured evidence record containing match metadata but
NO raw PII or biometric data.  Canonicalizes JSON deterministically
(sorted keys, compact separators) and computes SHA-256 for anchoring.
"""

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvidenceRecord:
    """Structured evidence — no raw biometrics, no PII unless consented."""
    confidence_tier: str
    fused_score: float
    face_similarity: float
    phash_similarity: float
    text_similarity: float
    discovered_url: str
    platform: str
    timestamp: str                        # ISO 8601 UTC
    candidate_image_sha256: str
    # PII fields — only populated when include_pii=True
    subject_name: Optional[str] = None
    subject_handle: Optional[str] = None


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

def canonicalize(evidence: Dict[str, Any]) -> str:
    """Deterministic JSON canonicalization.

    Uses sorted keys and compact separators (',', ':') so the same
    logical record always produces the exact same byte string, making
    SHA-256 hashing reproducible.

    Convention: None values are excluded from the canonical form to
    ensure PII-excluded and PII-included records don't accidentally
    collide.
    """
    # Strip None values for determinism
    cleaned = {k: v for k, v in evidence.items() if v is not None}
    return json.dumps(cleaned, sort_keys=True, separators=(",", ":"))


def evidence_hash(canonical_json: str) -> str:
    """Compute SHA-256 hex digest of the canonical JSON string."""
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_evidence(
    confidence_tier: str,
    fused_score: float,
    face_similarity: float,
    phash_similarity: float,
    text_similarity: float,
    discovered_url: str,
    platform: str,
    candidate_image_bytes: bytes,
    include_pii: bool = False,
    subject_name: Optional[str] = None,
    subject_handle: Optional[str] = None,
) -> EvidenceRecord:
    """Build a structured evidence record.

    Args:
        confidence_tier: "HIGH", "MEDIUM", or "LOW".
        fused_score: Final sigmoid-squashed confidence score.
        face_similarity: Individual face cosine similarity.
        phash_similarity: Individual pHash similarity.
        text_similarity: Individual text similarity.
        discovered_url: URL where the match was found.
        platform: Classified platform (e.g. "linkedin").
        candidate_image_bytes: Raw bytes of the downloaded candidate image.
        include_pii: Whether to include PII fields (requires explicit consent).
        subject_name: Name of subject (only stored if include_pii=True).
        subject_handle: Handle/username (only stored if include_pii=True).

    Returns:
        EvidenceRecord instance.
    """
    return EvidenceRecord(
        confidence_tier=confidence_tier,
        fused_score=round(fused_score, 4),
        face_similarity=round(face_similarity, 4),
        phash_similarity=round(phash_similarity, 4),
        text_similarity=round(text_similarity, 4),
        discovered_url=discovered_url,
        platform=platform,
        timestamp=datetime.now(timezone.utc).isoformat(),
        candidate_image_sha256=hashlib.sha256(candidate_image_bytes).hexdigest(),
        subject_name=subject_name if include_pii else None,
        subject_handle=subject_handle if include_pii else None,
    )


def evidence_to_dict(record: EvidenceRecord) -> Dict[str, Any]:
    """Convert EvidenceRecord to dict for canonicalization."""
    return asdict(record)


def compute_evidence_hash(record: EvidenceRecord) -> str:
    """Convenience: record → dict → canonical JSON → SHA-256."""
    d = evidence_to_dict(record)
    canonical = canonicalize(d)
    return evidence_hash(canonical)


def save_evidence(
    record: EvidenceRecord,
    hash_hex: str,
    path: str,
) -> None:
    """Save evidence record + hash to a local JSON file.

    This file is used for re-verification / tamper detection.
    """
    payload = {
        "evidence": evidence_to_dict(record),
        "evidence_hash": hash_hex,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_evidence(path: str) -> Dict[str, Any]:
    """Load a saved evidence file."""
    with open(path, "r") as f:
        return json.load(f)
