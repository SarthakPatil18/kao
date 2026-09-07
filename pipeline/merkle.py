"""
Stage 7a — Merkle Tree

Binary Merkle tree for batching evidence hashes.  Instead of anchoring one
hash per on-chain transaction, a batch of evidence records shares a single
Merkle root, cutting gas costs linearly.

Proof generation and verification are standard:
  - Proof = list of (sibling_hash, direction) pairs from leaf to root
  - Verify = reconstruct root from leaf + proof, compare to expected root
"""

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(data: str) -> str:
    """SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _hash_pair(left: str, right: str) -> str:
    """Hash two hex digests together (sorted to make tree order-independent
    at each level, following Ethereum convention)."""
    # Sort to ensure deterministic ordering
    if left > right:
        left, right = right, left
    return _sha256(left + right)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MerkleProof:
    """Proof of inclusion for a single leaf."""
    leaf: str
    proof: List[Tuple[str, str]]   # [(sibling_hash, "L"|"R"), ...]
    root: str


@dataclass
class MerkleTree:
    """Binary Merkle tree."""
    leaves: List[str]
    layers: List[List[str]] = field(default_factory=list)
    root: str = ""

    def get_proof(self, leaf_hash: str) -> Optional[MerkleProof]:
        """Generate a Merkle proof for a specific leaf.

        Args:
            leaf_hash: The leaf hash to prove inclusion for.

        Returns:
            MerkleProof if leaf exists in tree, None otherwise.
        """
        if leaf_hash not in self.leaves:
            return None

        idx = self.leaves.index(leaf_hash)
        proof_pairs: List[Tuple[str, str]] = []

        for layer in self.layers[:-1]:  # Skip the root layer
            # Sibling index
            if idx % 2 == 0:
                sibling_idx = idx + 1
                direction = "R"   # Sibling is on the right
            else:
                sibling_idx = idx - 1
                direction = "L"   # Sibling is on the left

            if sibling_idx < len(layer):
                proof_pairs.append((layer[sibling_idx], direction))
            else:
                # Odd leaf — promoted without sibling
                proof_pairs.append((layer[idx], "R"))

            idx //= 2

        return MerkleProof(leaf=leaf_hash, proof=proof_pairs, root=self.root)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_merkle_tree(leaf_hashes: List[str]) -> MerkleTree:
    """Build a binary Merkle tree from a list of leaf hashes.

    Works with any number of leaves (including 1).  Odd layers duplicate
    the last element to maintain a complete binary tree.

    Args:
        leaf_hashes: List of hex-encoded SHA-256 hashes.

    Returns:
        MerkleTree with root, layers, and proof generation.
    """
    if not leaf_hashes:
        return MerkleTree(leaves=[], layers=[], root="")

    # Single leaf — root IS the leaf
    if len(leaf_hashes) == 1:
        tree = MerkleTree(
            leaves=leaf_hashes,
            layers=[leaf_hashes],
            root=leaf_hashes[0],
        )
        return tree

    layers: List[List[str]] = [list(leaf_hashes)]
    current = list(leaf_hashes)

    while len(current) > 1:
        next_layer: List[str] = []
        for i in range(0, len(current), 2):
            left = current[i]
            right = current[i + 1] if i + 1 < len(current) else current[i]
            next_layer.append(_hash_pair(left, right))
        layers.append(next_layer)
        current = next_layer

    tree = MerkleTree(
        leaves=leaf_hashes,
        layers=layers,
        root=current[0],
    )
    return tree


def verify_merkle_proof(
    leaf_hash: str,
    proof: List[Tuple[str, str]],
    expected_root: str,
) -> bool:
    """Verify a Merkle proof by reconstructing the root.

    Args:
        leaf_hash: The leaf being proved.
        proof: List of (sibling_hash, direction) pairs.
        expected_root: The expected Merkle root to compare against.

    Returns:
        True if reconstructed root matches expected_root.
    """
    current = leaf_hash
    for sibling, direction in proof:
        if direction == "L":
            current = _hash_pair(sibling, current)
        else:
            current = _hash_pair(current, sibling)
    return current == expected_root
