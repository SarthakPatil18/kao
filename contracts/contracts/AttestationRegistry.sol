// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title AttestationRegistry
 * @notice Anchors Merkle roots of evidence batches on-chain.
 *
 * DESIGN PHILOSOPHY:
 *   This contract proves EVIDENCE INTEGRITY, not IDENTITY OWNERSHIP.
 *   It stores only a SHA-256 Merkle root (bytes32) and an IPFS CID pointer.
 *   No raw face images, embeddings, PII (names, emails, phone numbers),
 *   or biometric data are ever written on-chain.
 *
 *   The on-chain record answers: "This hash was submitted by address X at
 *   time T."  It does NOT answer: "Person Y owns account Z."
 */
contract AttestationRegistry {
    struct Anchor {
        uint256 timestamp;      // block.timestamp when anchored
        address submitter;      // address that submitted the root
        string ipfsCid;         // pointer to off-chain evidence batch manifest
        bool exists;            // true once recorded
    }

    /// @notice merkleRoot => Anchor
    mapping(bytes32 => Anchor) public anchors;

    /// @notice Emitted when a new Merkle root is anchored.
    event RootAnchored(
        bytes32 indexed merkleRoot,
        address submitter,
        string ipfsCid,
        uint256 timestamp
    );

    /**
     * @notice Anchor a Merkle root on-chain.
     * @param merkleRoot The SHA-256 Merkle root of the evidence batch.
     * @param ipfsCid    IPFS CID pointing to the evidence batch manifest.
     */
    function anchorRoot(bytes32 merkleRoot, string calldata ipfsCid) external {
        require(!anchors[merkleRoot].exists, "Root already anchored");
        anchors[merkleRoot] = Anchor(
            block.timestamp,
            msg.sender,
            ipfsCid,
            true
        );
        emit RootAnchored(merkleRoot, msg.sender, ipfsCid, block.timestamp);
    }

    /**
     * @notice Query an anchored record.
     * @param merkleRoot The Merkle root to look up.
     * @return timestamp  When the root was anchored.
     * @return submitter  Who submitted it.
     * @return ipfsCid    Off-chain evidence pointer.
     * @return exists     Whether the record exists.
     */
    function getAnchor(bytes32 merkleRoot)
        external
        view
        returns (
            uint256 timestamp,
            address submitter,
            string memory ipfsCid,
            bool exists
        )
    {
        Anchor memory a = anchors[merkleRoot];
        return (a.timestamp, a.submitter, a.ipfsCid, a.exists);
    }
}
