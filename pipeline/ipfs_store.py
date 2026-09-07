"""
Stage 6 — IPFS Storage

Uploads canonicalized evidence JSON to IPFS via Pinata for content-addressed
storage.  Returns a CID that inherently proves content integrity (the CID IS
a hash of the content).

Falls back gracefully to local storage if IPFS is unavailable.
"""

import json
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PINATA_PIN_URL = "https://api.pinata.cloud/pinning/pinJSONToIPFS"
PINATA_GATEWAY = "https://gateway.pinata.cloud/ipfs"
PUBLIC_GATEWAY = "https://ipfs.io/ipfs"
REQUEST_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Pinata IPFS upload
# ---------------------------------------------------------------------------

def upload_to_ipfs(
    evidence_json: str,
    pinata_jwt: str,
    metadata_name: str = "provenance-evidence",
) -> Optional[str]:
    """Upload evidence JSON to IPFS via Pinata.

    Args:
        evidence_json: Canonical JSON string of the evidence record.
        pinata_jwt: Pinata JWT authentication token.
        metadata_name: Name tag for the pinned content.

    Returns:
        IPFS CID (content identifier) on success, None on failure.
    """
    headers = {
        "Authorization": f"Bearer {pinata_jwt}",
        "Content-Type": "application/json",
    }

    # Pinata expects the JSON content under "pinataContent"
    payload = {
        "pinataOptions": {"cidVersion": 1},
        "pinataMetadata": {"name": metadata_name},
        "pinataContent": json.loads(evidence_json),
    }

    try:
        resp = requests.post(
            PINATA_PIN_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        cid = data.get("IpfsHash")
        if cid:
            logger.info(f"Evidence pinned to IPFS: {cid}")
            return cid
        else:
            logger.error(f"Pinata response missing IpfsHash: {data}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"IPFS upload failed: {e}")
        return None


# ---------------------------------------------------------------------------
# IPFS fetch
# ---------------------------------------------------------------------------

def fetch_from_ipfs(
    cid: str,
    pinata_jwt: Optional[str] = None,
) -> Optional[str]:
    """Retrieve evidence JSON from IPFS by CID.

    Tries Pinata gateway first (authenticated), then public gateway.

    Args:
        cid: IPFS content identifier.
        pinata_jwt: Optional Pinata JWT for authenticated gateway access.

    Returns:
        Raw JSON string on success, None on failure.
    """
    urls_to_try = []

    if pinata_jwt:
        urls_to_try.append(
            (f"{PINATA_GATEWAY}/{cid}", {"Authorization": f"Bearer {pinata_jwt}"})
        )

    urls_to_try.append((f"{PUBLIC_GATEWAY}/{cid}", {}))

    for url, headers in urls_to_try:
        try:
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            content = resp.text
            # Validate it's JSON
            json.loads(content)
            logger.info(f"Fetched evidence from IPFS: {cid}")
            return content
        except Exception as e:
            logger.warning(f"Failed to fetch from {url}: {e}")
            continue

    logger.error(f"Could not fetch CID {cid} from any gateway")
    return None


# ---------------------------------------------------------------------------
# Local fallback
# ---------------------------------------------------------------------------

def save_locally(evidence_json: str, path: str) -> str:
    """Save evidence JSON to local file as IPFS fallback.

    Returns the file path (acting as a pseudo-CID).
    """
    with open(path, "w") as f:
        f.write(evidence_json)
    logger.info(f"Evidence saved locally (IPFS fallback): {path}")
    return path


def fetch_locally(path: str) -> Optional[str]:
    """Read evidence JSON from local file."""
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Local evidence file not found: {path}")
        return None


# ---------------------------------------------------------------------------
# Unified store/fetch
# ---------------------------------------------------------------------------

def store_evidence(
    evidence_json: str,
    pinata_jwt: Optional[str] = None,
    local_fallback_path: Optional[str] = None,
) -> Optional[str]:
    """Store evidence, preferring IPFS with local fallback.

    Returns:
        CID if IPFS succeeded, local path if fallback used, None if all failed.
    """
    if pinata_jwt:
        cid = upload_to_ipfs(evidence_json, pinata_jwt)
        if cid:
            return cid

    if local_fallback_path:
        return save_locally(evidence_json, local_fallback_path)

    logger.error("No storage method available (no IPFS key, no local path)")
    return None
