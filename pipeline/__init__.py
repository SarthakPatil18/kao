"""
PROVENANCE — Consent-Gated Face Discovery & Merkle-Anchored Attestation Pipeline

This package implements the core pipeline stages:
  1. liveness   — MediaPipe FaceMesh liveness check (hard gate)
  2. face       — InsightFace detection + 512-d embedding
  3. discovery  — SerpApi / Bing reverse image search
  4. scoring    — Multi-signal sigmoid fusion scoring
  5. evidence   — Canonicalized evidence record + SHA-256
  6. ipfs_store — IPFS upload/fetch via Pinata
  7. merkle     — Binary Merkle tree with proof generation
  8. blockchain — Polygon Amoy on-chain anchoring
  9. verify     — Re-verification + tamper detection
"""

__version__ = "0.1.0"
