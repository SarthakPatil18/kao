"""
Stage 8 — Re-Verification & Tamper Detection

Re-verification workflow:
  1. Fetch evidence from IPFS (or local file) by CID.
  2. Recompute its SHA-256 hash locally.
  3. Verify the recomputed hash matches the original leaf hash.
  4. Walk the Merkle proof to reconstruct the root.
  5. Compare reconstructed root to the on-chain root.
  6. Return VERIFIED or TAMPER_DETECTED with detailed diff.

Tamper demo:
  Mutate one field in the evidence, recompute hash, show it no longer
  matches the proof — proving the system detects alterations.
"""

import copy
import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from pipeline.evidence import canonicalize, evidence_hash
from pipeline.merkle import verify_merkle_proof
from pipeline.ipfs_store import fetch_from_ipfs, fetch_locally

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    """Result of evidence re-verification."""
    status: str                              # "VERIFIED" or "TAMPER_DETECTED"
    evidence_hash_match: bool                # Recomputed hash matches leaf?
    merkle_proof_valid: bool                 # Proof reconstructs to root?
    onchain_root_match: bool                 # Reconstructed root matches chain?
    recomputed_hash: str
    expected_hash: str
    details: str                             # Human-readable explanation


@dataclass
class TamperResult:
    """Result of a tamper demonstration."""
    original_hash: str
    tampered_hash: str
    tampered_field: str
    original_value: str
    tampered_value: str
    hashes_match: bool                       # Should be False
    proof_valid: bool                        # Should be False
    explanation: str


# ---------------------------------------------------------------------------
# Re-verification
# ---------------------------------------------------------------------------

def reverify(
    evidence_source: str,
    expected_leaf_hash: str,
    merkle_proof: List[Tuple[str, str]],
    expected_root: str,
    onchain_root: Optional[str] = None,
    pinata_jwt: Optional[str] = None,
) -> VerificationResult:
    """Re-verify an evidence record end-to-end.

    Args:
        evidence_source: IPFS CID or local file path.
        expected_leaf_hash: The original SHA-256 hash of the evidence.
        merkle_proof: List of (sibling_hash, direction) for proof walk.
        expected_root: Expected Merkle root.
        onchain_root: Root retrieved from on-chain (if available).
        pinata_jwt: Pinata JWT for IPFS gateway access.

    Returns:
        VerificationResult with status and details.
    """
    # Step 1: Fetch evidence
    evidence_json = None

    # Try IPFS first
    if evidence_source and not evidence_source.startswith("/"):
        evidence_json = fetch_from_ipfs(evidence_source, pinata_jwt)

    # Fallback to local
    if evidence_json is None:
        evidence_json = fetch_locally(evidence_source)

    if evidence_json is None:
        return VerificationResult(
            status="TAMPER_DETECTED",
            evidence_hash_match=False,
            merkle_proof_valid=False,
            onchain_root_match=False,
            recomputed_hash="",
            expected_hash=expected_leaf_hash,
            details=f"Could not retrieve evidence from: {evidence_source}",
        )

    # Step 2: Recompute hash
    # Parse the JSON, re-canonicalize, re-hash
    try:
        evidence_dict = json.loads(evidence_json)
        # If wrapped in {"evidence": ..., "evidence_hash": ...} format
        if "evidence" in evidence_dict:
            evidence_dict = evidence_dict["evidence"]
    except json.JSONDecodeError as e:
        return VerificationResult(
            status="TAMPER_DETECTED",
            evidence_hash_match=False,
            merkle_proof_valid=False,
            onchain_root_match=False,
            recomputed_hash="",
            expected_hash=expected_leaf_hash,
            details=f"Evidence JSON is malformed: {e}",
        )

    canonical = canonicalize(evidence_dict)
    recomputed = evidence_hash(canonical)

    # Step 3: Compare hashes
    hash_match = recomputed == expected_leaf_hash

    # Step 4: Verify Merkle proof
    proof_valid = verify_merkle_proof(
        expected_leaf_hash, merkle_proof, expected_root
    )

    # Step 5: Compare to on-chain root
    if onchain_root is not None:
        chain_match = expected_root == onchain_root
    else:
        chain_match = True  # No on-chain check available

    # Determine status
    if hash_match and proof_valid and chain_match:
        status = "VERIFIED"
        details = (
            "Evidence integrity confirmed. "
            "Recomputed hash matches the original, "
            "Merkle proof is valid, and on-chain root matches."
        )
    else:
        status = "TAMPER_DETECTED"
        issues = []
        if not hash_match:
            issues.append(
                f"Hash mismatch: recomputed={recomputed[:16]}... "
                f"expected={expected_leaf_hash[:16]}..."
            )
        if not proof_valid:
            issues.append("Merkle proof reconstruction does not match root.")
        if not chain_match:
            issues.append(
                f"On-chain root mismatch: expected={expected_root[:16]}... "
                f"on-chain={onchain_root[:16] if onchain_root else 'N/A'}..."
            )
        details = "TAMPER DETECTED. " + " | ".join(issues)

    return VerificationResult(
        status=status,
        evidence_hash_match=hash_match,
        merkle_proof_valid=proof_valid,
        onchain_root_match=chain_match,
        recomputed_hash=recomputed,
        expected_hash=expected_leaf_hash,
        details=details,
    )


# ---------------------------------------------------------------------------
# Tamper demonstration
# ---------------------------------------------------------------------------

def tamper_demo(
    evidence_dict: dict,
    original_hash: str,
    merkle_proof: List[Tuple[str, str]],
    merkle_root: str,
    field_to_mutate: str = "confidence_tier",
    mutated_value: str = "HIGH_TAMPERED",
) -> TamperResult:
    """Demonstrate tamper detection by mutating a field.

    Args:
        evidence_dict: Original evidence dictionary.
        original_hash: Original SHA-256 hash of the evidence.
        merkle_proof: Original Merkle proof.
        merkle_root: Original Merkle root.
        field_to_mutate: Which field to tamper with.
        mutated_value: What to change it to.

    Returns:
        TamperResult showing the mismatch.
    """
    # Create a tampered copy
    tampered = copy.deepcopy(evidence_dict)
    original_value = str(tampered.get(field_to_mutate, "N/A"))
    tampered[field_to_mutate] = mutated_value

    # Recompute hash of tampered evidence
    tampered_canonical = canonicalize(tampered)
    tampered_hash = evidence_hash(tampered_canonical)

    # Check if proof still holds (it shouldn't)
    proof_valid = verify_merkle_proof(tampered_hash, merkle_proof, merkle_root)

    hashes_match = tampered_hash == original_hash

    explanation = (
        f"Field '{field_to_mutate}' was mutated from '{original_value}' "
        f"to '{mutated_value}'. "
        f"Original hash: {original_hash[:20]}... | "
        f"Tampered hash: {tampered_hash[:20]}... | "
        f"Hashes match: {hashes_match} | "
        f"Merkle proof valid: {proof_valid} | "
        f"{'⚠️ TAMPER DETECTED — evidence has been altered.' if not hashes_match else '❌ ERROR: tampered evidence should not match!'}"
    )

    return TamperResult(
        original_hash=original_hash,
        tampered_hash=tampered_hash,
        tampered_field=field_to_mutate,
        original_value=original_value,
        tampered_value=mutated_value,
        hashes_match=hashes_match,
        proof_valid=proof_valid,
        explanation=explanation,
    )
