"""
Stage 7b — Blockchain Anchoring

Anchors a Merkle root (+ IPFS CID pointer) on Polygon Amoy testnet via
the AttestationRegistry smart contract.

SECURITY:
  - Private key loaded from .env ONLY — never logged, never committed.
  - Use a throwaway testnet-only wallet.
  - No raw biometric or PII data ever goes on-chain.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

try:
    from web3 import Web3
    from web3.middleware import ExtraDataToPOAMiddleware
except ImportError:
    Web3 = None
    ExtraDataToPOAMiddleware = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Contract ABI (minimal — only the functions we call)
# ---------------------------------------------------------------------------

CONTRACT_ABI = json.loads("""[
    {
        "inputs": [
            {"internalType": "bytes32", "name": "merkleRoot", "type": "bytes32"},
            {"internalType": "string", "name": "ipfsCid", "type": "string"}
        ],
        "name": "anchorRoot",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "merkleRoot", "type": "bytes32"}
        ],
        "name": "getAnchor",
        "outputs": [
            {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
            {"internalType": "address", "name": "submitter", "type": "address"},
            {"internalType": "string", "name": "ipfsCid", "type": "string"},
            {"internalType": "bool", "name": "exists", "type": "bool"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "anonymous": false,
        "inputs": [
            {"indexed": true, "internalType": "bytes32", "name": "merkleRoot", "type": "bytes32"},
            {"indexed": false, "internalType": "address", "name": "submitter", "type": "address"},
            {"indexed": false, "internalType": "string", "name": "ipfsCid", "type": "string"},
            {"indexed": false, "internalType": "uint256", "name": "timestamp", "type": "uint256"}
        ],
        "name": "RootAnchored",
        "type": "event"
    }
]""")

# Default Polygon Amoy RPC
DEFAULT_RPC = "https://rpc-amoy.polygon.technology/"
AMOY_CHAIN_ID = 80002
AMOY_EXPLORER = "https://amoy.polygonscan.com"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AnchorResult:
    """Result of querying an on-chain anchor."""
    exists: bool
    timestamp: int
    submitter: str
    ipfs_cid: str


@dataclass
class TransactionResult:
    """Result of an anchoring transaction."""
    success: bool
    tx_hash: str
    block_number: int
    gas_used: int
    explorer_url: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def connect(rpc_url: Optional[str] = None) -> "Web3":
    """Connect to the blockchain via JSON-RPC.

    Args:
        rpc_url: RPC endpoint URL.  Defaults to Polygon Amoy public RPC.

    Returns:
        Connected Web3 instance.
    """
    if Web3 is None:
        raise ImportError("web3 not installed. Run: pip install web3")

    url = rpc_url or DEFAULT_RPC
    w3 = Web3(Web3.HTTPProvider(url))

    # Polygon Amoy is a PoA chain — need middleware for extraData
    if ExtraDataToPOAMiddleware is not None:
        try:
            w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        except Exception:
            pass  # Already injected or not needed

    if not w3.is_connected():
        raise ConnectionError(f"Cannot connect to {url}")

    logger.info(f"Connected to chain: {url}")
    return w3


def get_contract(w3: "Web3", contract_address: str):
    """Get a contract instance.

    Args:
        w3: Connected Web3 instance.
        contract_address: Deployed contract address.

    Returns:
        Web3 Contract instance.
    """
    return w3.eth.contract(
        address=Web3.to_checksum_address(contract_address),
        abi=CONTRACT_ABI,
    )


# ---------------------------------------------------------------------------
# Anchoring
# ---------------------------------------------------------------------------

def anchor_root(
    w3: "Web3",
    contract,
    merkle_root: str,
    ipfs_cid: str,
    private_key: str,
) -> TransactionResult:
    """Anchor a Merkle root on-chain.

    Args:
        w3: Connected Web3 instance.
        contract: AttestationRegistry contract instance.
        merkle_root: Hex-encoded Merkle root (64 chars, no 0x prefix).
        ipfs_cid: IPFS CID pointing to the evidence batch manifest.
        private_key: Wallet private key (from .env, never logged).

    Returns:
        TransactionResult with tx hash and status.
    """
    try:
        account = w3.eth.account.from_key(private_key)
        root_bytes = bytes.fromhex(merkle_root)

        # Build transaction
        tx = contract.functions.anchorRoot(root_bytes, ipfs_cid).build_transaction({
            "from": account.address,
            "nonce": w3.eth.get_transaction_count(account.address),
            "gas": 200_000,
            "gasPrice": w3.eth.gas_price,
            "chainId": AMOY_CHAIN_ID,
        })

        # Sign and send
        signed = w3.eth.account.sign_transaction(tx, private_key)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

        tx_hex = receipt.transactionHash.hex()
        explorer_url = f"{AMOY_EXPLORER}/tx/0x{tx_hex}"

        logger.info(f"Root anchored on-chain: {explorer_url}")

        return TransactionResult(
            success=receipt.status == 1,
            tx_hash=f"0x{tx_hex}",
            block_number=receipt.blockNumber,
            gas_used=receipt.gasUsed,
            explorer_url=explorer_url,
            error=None if receipt.status == 1 else "Transaction reverted",
        )

    except Exception as e:
        logger.error(f"Anchoring failed: {e}")
        return TransactionResult(
            success=False,
            tx_hash="",
            block_number=0,
            gas_used=0,
            explorer_url="",
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------

def get_anchor(
    w3: "Web3",
    contract,
    merkle_root: str,
) -> AnchorResult:
    """Query an on-chain anchor by Merkle root.

    Args:
        w3: Connected Web3 instance.
        contract: AttestationRegistry contract instance.
        merkle_root: Hex-encoded Merkle root (64 chars, no 0x prefix).

    Returns:
        AnchorResult with on-chain data.
    """
    root_bytes = bytes.fromhex(merkle_root)
    timestamp, submitter, ipfs_cid, exists = contract.functions.getAnchor(
        root_bytes
    ).call()

    return AnchorResult(
        exists=exists,
        timestamp=timestamp,
        submitter=submitter,
        ipfs_cid=ipfs_cid,
    )


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def load_blockchain_config() -> dict:
    """Load blockchain config from .env file.

    Returns:
        Dict with rpc_url, private_key, contract_address.
    """
    load_dotenv()
    return {
        "rpc_url": os.getenv("RPC_URL", DEFAULT_RPC),
        "private_key": os.getenv("PRIVATE_KEY", ""),
        "contract_address": os.getenv("CONTRACT_ADDRESS", ""),
    }
