# KAO (顔)

### Consent-Gated Face Discovery & Merkle-Anchored Attestation Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-black?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-black?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-black?style=flat-square&logo=react)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-5-black?style=flat-square&logo=vite)](https://vitejs.dev)
[![Polygon](https://img.shields.io/badge/Polygon-Amoy_Testnet-black?style=flat-square&logo=polygon)](https://amoy.polygonscan.com)
[![IPFS](https://img.shields.io/badge/IPFS-Pinata-black?style=flat-square&logo=ipfs)](https://pinata.cloud)
[![Tests](https://img.shields.io/badge/Tests-65%2F65_Passed-black?style=flat-square)](tests/)

> **Evidence integrity, not identity ownership.**
> KAO proves that an evidence record captured at timestamp $T$ has not been modified or tampered with since anchoring. It does **not** claim to prove that *"Person X is the legal owner of Account Y"*.

---

## 1. System Architecture

```mermaid
flowchart TD
    subgraph Client ["Client Interface (React + Vite)"]
        UI_Input["Photo Upload / Live Camera"]
        UI_Consent{"Consent Confirmed?"}
        UI_Feed["Real-Time Status Feed"]
        UI_Result["Result Card & Tamper Diff"]
    end

    subgraph Backend ["FastAPI Pipeline Service"]
        API_Gate["Consent Enforcement Gate"]
        Liveness["Stage 1: Liveness Challenge"]
        InsightFace["Stage 2: RetinaFace + ArcFace (512-d)"]
        Discovery["Stage 3: Google Lens / Bing Visual"]
        Scoring["Stage 4: Multi-Signal Scoring"]
        Evidence["Stage 5: Canonical JSON (RFC 8785)"]
        Merkle["Stage 6: Merkle Tree & Binary Proof"]
    end

    subgraph Storage_Chain ["Decentralized Infrastructure"]
        IPFS["IPFS / Pinata (CID Storage)"]
        Polygon["Polygon Amoy Testnet (EvidenceRegistry.sol)"]
    end

    subgraph Verification ["Cryptographic Auditing"]
        Reverify["Re-Verification: SHA-256 Recalculation"]
        Tamper["Tamper Test: Sibling Proof Invalidation"]
    end

    UI_Input --> UI_Consent
    UI_Consent -- Yes --> API_Gate
    UI_Consent -- No --> UI_Input
    API_Gate --> Liveness
    Liveness --> InsightFace
    InsightFace --> Discovery
    Discovery --> Scoring
    Scoring --> Evidence
    Evidence --> Merkle
    Evidence --> IPFS
    Merkle --> Polygon
    IPFS -. CID .-> Polygon
    Polygon --> UI_Feed
    UI_Feed --> UI_Result
    UI_Result --> Tamper
    Tamper --> Reverify
```

---

## 2. End-to-End Sequence Flow

```mermaid
sequenceDiagram
    autonumber
    actor User as Consenting Subject
    participant Frontend as Vite / React SPA
    participant Backend as FastAPI Server
    participant CV as InsightFace / ONNX
    participant Search as SerpApi / Google Lens
    participant IPFS as Pinata IPFS Gateway
    participant Chain as Polygon Amoy (Web3)

    User->>Frontend: Selects photo or captures live frame
    User->>Frontend: Ticks consent confirmation
    User->>Frontend: Clicks "Verify"
    Frontend->>Backend: POST /api/pipeline/execute (Multipart)
    Backend->>Backend: Verify Subject Consent (403 if missing)
    Backend->>CV: Face Detection (det_10g) & Embedding (w600k_r50)
    CV-->>Backend: 512-d feature vector + Quality Score
    Backend->>Search: Query Reverse Image Search
    Search-->>Backend: Ranked Candidate Matches & Platform Metadata
    Backend->>Backend: Compute Multi-Signal Fusion (Face + pHash + Text)
    Backend->>Backend: Generate Canonical JSON (RFC 8785) & SHA-256 Digest
    Backend->>Backend: Build Merkle Tree & Binary Path Proof
    Backend->>IPFS: Upload Evidence JSON (Receive CID)
    Backend->>Chain: anchorRoot(merkleRoot, ipfsCid)
    Chain-->>Backend: Transaction Hash & Block Receipt
    Backend-->>Frontend: 200 OK (Full Pipeline Payload)
    Frontend->>User: Display Result Card + Live Status + On-chain Proofs
```

---

## 3. The 9-Stage Verification & Attestation Pipeline

| Stage | Name | Technology | Security & Privacy Guarantees |
|---|---|---|---|
| **0** | **Consent Gate** | UI State + Backend Guard | Pipeline rejects requests without explicit subject authorization (**HTTP 403**). |
| **1** | **Liveness Check** | Interactive Challenge | Prevents automated scraping of static third-party photographs. |
| **2** | **Face Extraction** | InsightFace RetinaFace + ArcFace | Computes 512-d biometric embedding in volatile RAM. **Never persisted to disk or chain.** |
| **3** | **Web Discovery** | SerpApi (Google Lens) / Bing | Multi-platform discovery (LinkedIn, GitHub, Twitter/X, Medium, News). |
| **4** | **Multi-Signal Scoring** | Cosine Sim + pHash + TF-IDF | Fuses face similarity ($0.50$), perceptual hash ($0.25$), text similarity ($0.15$), and rank ($0.10$). |
| **5** | **Evidence Canonicalization**| RFC 8785 JSON + SHA-256 | Deterministic byte-for-byte serialization producing a immutable leaf hash. |
| **6** | **Merkle Batching** | Binary Merkle Tree | Computes cryptographically verifiable inclusion proofs `[(sibling, direction)]`. |
| **7a**| **Decentralized Storage** | Pinata IPFS | Content-addressed storage returning a tamper-resistant IPFS CID. |
| **7b**| **Blockchain Anchoring** | Polygon Amoy / Web3.py | Smart contract stores `(merkleRoot, ipfsCid, timestamp, submitter)` on-chain. |
| **8** | **Re-Verification & Tampering**| Cryptographic Audit | Reconstructs the Merkle root from leaf to proof; detects alteration of even 1 bit. |

---

## 4. Merkle Tree & Tamper-Detection Architecture

```mermaid
graph TD
    Root["Merkle Root (Anchored on Polygon Amoy)"]
    H12["Node H12 = SHA256(H1 + H2)"]
    H34["Node H34 = SHA256(H3 + H4)"]
    L1["Leaf 1: Target Evidence (SHA256)"]
    L2["Leaf 2: Candidate B"]
    L3["Leaf 3: Candidate C"]
    L4["Leaf 4: Candidate D"]

    Root --- H12
    Root --- H34
    H12 --- L1
    H12 --- L2
    H34 --- L3
    H34 --- L4

    classDef target fill:#000,stroke:#fff,stroke-width:2px,color:#fff;
    classDef sibling fill:#333,stroke:#888,stroke-width:1px,color:#eee;
    class L1 target;
    class L2,H34 sibling;
```

When a user triggers the **Run Tamper Test**:
1. An evidence field (e.g. `confidence_tier`) is modified from `HIGH` to `LOW`.
2. The leaf hash is recomputed: $H_{\text{tampered}} \neq H_{\text{original}}$.
3. Walking the Merkle proof produces $R_{\text{tampered}} \neq R_{\text{anchored}}$.
4. Cryptographic audit fails immediately with `hashes_match: false` and `proof_valid: false`.

---

## 5. Smart Contract: `EvidenceRegistry.sol`

Deployed to the **Polygon Amoy Testnet** (Chain ID `80002`):

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract EvidenceRegistry {
    struct AnchorRecord {
        uint256 timestamp;
        address submitter;
        string ipfsCid;
        bool exists;
    }

    mapping(bytes32 => AnchorRecord) public anchors;

    event RootAnchored(
        bytes32 indexed merkleRoot,
        address submitter,
        string ipfsCid,
        uint256 timestamp
    );

    function anchorRoot(bytes32 merkleRoot, string calldata ipfsCid) external {
        require(!anchors[merkleRoot].exists, "Root already anchored");
        anchors[merkleRoot] = AnchorRecord({
            timestamp: block.timestamp,
            submitter: msg.sender,
            ipfsCid: ipfsCid,
            exists: true
        });
        emit RootAnchored(merkleRoot, msg.sender, ipfsCid, block.timestamp);
    }

    function getAnchor(bytes32 merkleRoot) external view returns (AnchorRecord memory) {
        return anchors[merkleRoot];
    }
}
```

---

## 6. Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+ & npm
- macOS / Linux / WSL

### 1. Clone & Setup Repository
```bash
git clone https://github.com/SarthakPatil18/kao.git
cd kao
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Populate `.env` with your API keys (optional — offline simulation works out-of-the-box):
```env
SERPAPI_KEY=your_serpapi_key_here
PINATA_JWT=your_pinata_jwt_here
RPC_URL=https://rpc-amoy.polygon.technology/
PRIVATE_KEY=your_testnet_private_key_here
CONTRACT_ADDRESS=your_contract_address_here
```

### 3. Install Backend Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

---

## 7. Running the Application

### Start the FastAPI Backend (Port 8000)
```bash
python3 -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```
* **API URL**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Interactive OpenAPI Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Start the React Frontend (Port 3000)
In a separate terminal:
```bash
cd frontend
npm run dev -- --host 127.0.0.1 --port 3000
```
* **Web UI**: [http://localhost:3000](http://localhost:3000)

---

## 8. Verification & Test Suite

Run the automated test suite covering all cryptographic, biometric, and Merkle operations:

```bash
pytest tests/ -v
```

```text
============================= test session starts ==============================
collected 65 items

tests/test_evidence.py ............                                      [ 18%]
tests/test_liveness.py ..............                                    [ 40%]
tests/test_merkle.py ...............                                     [ 63%]
tests/test_scoring.py ........................                           [100%]

============================== 65 passed in 10.9s ==============================
```

---

## 9. Security & Privacy Philosophy

1. **Zero Biometric Persistence**: Raw face images and 512-d feature vectors exist only in volatile process memory during execution and are discarded immediately.
2. **Deterministic Evidence Proofs**: RFC 8785 canonical JSON ensures that whitespace or key-ordering variations never cause false tamper alarms.
3. **Strict Consent Policy**: The consent checkbox is a hard blocker at both frontend and API levels.
4. **Minimal On-Chain Footprint**: Only cryptographic root hashes and content identifiers touch public ledgers.

---

## 10. License

Licensed under the [MIT License](LICENSE).
