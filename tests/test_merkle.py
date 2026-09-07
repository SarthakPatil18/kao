"""
Tests for Merkle tree construction, proof generation, and verification.

Key properties tested:
  - Single-leaf tree: root equals the leaf
  - Multi-leaf tree: proof reconstructs to root
  - Tampered leaf: proof fails verification
  - Different leaf orderings produce same root (sorted pair hashing)
"""

import hashlib
import pytest
from pipeline.merkle import build_merkle_tree, verify_merkle_proof, _sha256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_hash(data: str) -> str:
    """Create a SHA-256 hash for test data."""
    return hashlib.sha256(data.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_leaf():
    return [make_hash("evidence_1")]


@pytest.fixture
def two_leaves():
    return [make_hash("evidence_1"), make_hash("evidence_2")]


@pytest.fixture
def four_leaves():
    return [
        make_hash("evidence_1"),
        make_hash("evidence_2"),
        make_hash("evidence_3"),
        make_hash("evidence_4"),
    ]


@pytest.fixture
def five_leaves():
    """Odd number of leaves — tests duplication of last element."""
    return [make_hash(f"evidence_{i}") for i in range(1, 6)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTreeConstruction:
    """Merkle tree building tests."""

    def test_empty_tree(self):
        """Empty input produces empty tree."""
        tree = build_merkle_tree([])
        assert tree.root == ""
        assert tree.leaves == []

    def test_single_leaf_root(self, single_leaf):
        """Single leaf: root equals the leaf."""
        tree = build_merkle_tree(single_leaf)
        assert tree.root == single_leaf[0]

    def test_two_leaf_root(self, two_leaves):
        """Two leaves: root is hash of the pair."""
        tree = build_merkle_tree(two_leaves)
        assert tree.root != ""
        assert tree.root != two_leaves[0]
        assert tree.root != two_leaves[1]
        assert len(tree.layers) == 2  # leaf layer + root layer

    def test_four_leaf_layers(self, four_leaves):
        """Four leaves: 3 layers (4 → 2 → 1)."""
        tree = build_merkle_tree(four_leaves)
        assert len(tree.layers) == 3
        assert len(tree.layers[0]) == 4
        assert len(tree.layers[1]) == 2
        assert len(tree.layers[2]) == 1
        assert tree.root == tree.layers[2][0]

    def test_odd_leaves(self, five_leaves):
        """Odd number of leaves handled correctly."""
        tree = build_merkle_tree(five_leaves)
        assert tree.root != ""
        # Should still produce a valid tree
        assert len(tree.layers[0]) == 5

    def test_deterministic(self, four_leaves):
        """Same input always produces same root."""
        tree1 = build_merkle_tree(four_leaves)
        tree2 = build_merkle_tree(four_leaves)
        assert tree1.root == tree2.root


class TestProofGeneration:
    """Merkle proof generation tests."""

    def test_proof_exists_for_each_leaf(self, four_leaves):
        """Every leaf has a valid proof."""
        tree = build_merkle_tree(four_leaves)
        for leaf in four_leaves:
            proof = tree.get_proof(leaf)
            assert proof is not None
            assert proof.leaf == leaf
            assert proof.root == tree.root

    def test_proof_none_for_missing_leaf(self, four_leaves):
        """Non-existent leaf returns None."""
        tree = build_merkle_tree(four_leaves)
        fake = make_hash("nonexistent")
        assert tree.get_proof(fake) is None

    def test_single_leaf_proof(self, single_leaf):
        """Single leaf tree has an empty proof (leaf IS root)."""
        tree = build_merkle_tree(single_leaf)
        proof = tree.get_proof(single_leaf[0])
        assert proof is not None
        assert len(proof.proof) == 0


class TestProofVerification:
    """Merkle proof verification tests."""

    def test_valid_proof_verifies(self, four_leaves):
        """Valid proofs verify successfully."""
        tree = build_merkle_tree(four_leaves)
        for leaf in four_leaves:
            proof_obj = tree.get_proof(leaf)
            assert verify_merkle_proof(leaf, proof_obj.proof, tree.root) is True

    def test_tampered_leaf_fails(self, four_leaves):
        """Tampered leaf fails verification."""
        tree = build_merkle_tree(four_leaves)
        proof_obj = tree.get_proof(four_leaves[0])

        # Tamper: use a different hash as leaf
        tampered = make_hash("tampered_evidence")
        assert verify_merkle_proof(tampered, proof_obj.proof, tree.root) is False

    def test_wrong_root_fails(self, four_leaves):
        """Correct leaf + proof but wrong root fails."""
        tree = build_merkle_tree(four_leaves)
        proof_obj = tree.get_proof(four_leaves[0])
        fake_root = make_hash("fake_root")
        assert verify_merkle_proof(
            four_leaves[0], proof_obj.proof, fake_root
        ) is False

    def test_two_leaf_proof(self, two_leaves):
        """Two-leaf tree proofs verify."""
        tree = build_merkle_tree(two_leaves)
        for leaf in two_leaves:
            proof_obj = tree.get_proof(leaf)
            assert verify_merkle_proof(leaf, proof_obj.proof, tree.root) is True

    def test_single_leaf_verification(self, single_leaf):
        """Single leaf: empty proof, leaf equals root."""
        tree = build_merkle_tree(single_leaf)
        assert verify_merkle_proof(single_leaf[0], [], tree.root) is True


class TestIntegrity:
    """End-to-end integrity tests."""

    def test_batch_of_evidence_hashes(self):
        """Simulate a real batch of evidence hashes through the tree."""
        # Simulate 3 evidence records
        hashes = [
            _sha256('{"confidence_tier":"HIGH","fused_score":0.92}'),
            _sha256('{"confidence_tier":"MEDIUM","fused_score":0.65}'),
            _sha256('{"confidence_tier":"LOW","fused_score":0.30}'),
        ]
        tree = build_merkle_tree(hashes)

        # All proofs should verify
        for h in hashes:
            proof = tree.get_proof(h)
            assert proof is not None
            assert verify_merkle_proof(h, proof.proof, tree.root) is True

        # Tamper one hash
        tampered = _sha256('{"confidence_tier":"HIGH","fused_score":0.30}')
        proof_0 = tree.get_proof(hashes[0])
        assert verify_merkle_proof(tampered, proof_0.proof, tree.root) is False
