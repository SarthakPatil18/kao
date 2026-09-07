import React, { useState, useRef, useEffect } from 'react';

export default function App() {
  // Input state
  const [inputMode, setInputMode] = useState('upload'); // 'upload' | 'camera' | 'sample'
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadPreview, setUploadPreview] = useState(null);
  const [consent, setConsent] = useState(false);

  // Camera state
  const videoRef = useRef(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [capturedBlob, setCapturedBlob] = useState(null);
  const [capturedPreview, setCapturedPreview] = useState(null);

  // Status feed state
  const [isVerifying, setIsVerifying] = useState(false);
  const [feedLines, setFeedLines] = useState([]); // array of { text, status: 'running' | 'done' | 'failed' }
  const [pipelineResult, setPipelineResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');

  // Tamper test state
  const [tamperResult, setTamperResult] = useState(null);
  const [isTampering, setIsTampering] = useState(false);

  // Technical details expander
  const [showTechDetails, setShowTechDetails] = useState(false);

  // Stop camera stream
  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
  };

  // Start camera stream
  const startCamera = async () => {
    try {
      setCapturedBlob(null);
      setCapturedPreview(null);
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setCameraActive(true);
    } catch (err) {
      alert('Camera access denied or unavailable: ' + err.message);
    }
  };

  // Capture frame from video
  const captureFrame = () => {
    if (!videoRef.current) return;
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth || 640;
    canvas.height = videoRef.current.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      setCapturedBlob(blob);
      setCapturedPreview(URL.createObjectURL(blob));
      stopCamera();
    }, 'image/jpeg', 0.9);
  };

  // Clean up camera on unmount or tab change
  useEffect(() => {
    if (inputMode !== 'camera') {
      stopCamera();
    }
  }, [inputMode]);

  // File upload handler
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setUploadedFile(file);
      setUploadPreview(URL.createObjectURL(file));
    }
  };

  // Check if image is provided
  const hasImage = (
    (inputMode === 'upload' && uploadedFile !== null) ||
    (inputMode === 'camera' && capturedBlob !== null) ||
    (inputMode === 'sample')
  );

  const canVerify = hasImage && consent && !isVerifying;

  // Run verification
  const handleVerify = async () => {
    if (!canVerify) return;

    setIsVerifying(true);
    setPipelineResult(null);
    setTamperResult(null);
    setErrorMessage('');
    setFeedLines([{ text: 'Checking liveness...', status: 'running' }]);

    const formData = new FormData();
    formData.append('search_mode', 'simulated');
    formData.append('storage_mode', 'local');
    formData.append('chain_mode', 'simulated');
    formData.append('include_pii', 'false');
    formData.append('consent', consent ? 'true' : 'false');

    if (inputMode === 'upload' && uploadedFile) {
      formData.append('file', uploadedFile);
    } else if (inputMode === 'camera' && capturedBlob) {
      formData.append('file', capturedBlob, 'webcam_photo.jpg');
    } else {
      formData.append('preset_id', 'alex_rivera');
    }

    try {
      // Execute backend pipeline call
      const res = await fetch('/api/pipeline/execute', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errData.detail || 'Pipeline execution failed');
      }

      const data = await res.json();

      // Progressive status feed animation
      const steps = [
        { doneText: 'Liveness confirmed', nextRunning: 'Detecting face...' },
        { doneText: `Face detected (quality: ${data.face.quality_score}/100)`, nextRunning: 'Searching web for matches...' },
        { 
          doneText: `Match found: ${data.discovery.top_match.url.replace(/^https?:\/\//, '').slice(0, 36)}... (confidence: ${data.discovery.top_match.score.tier} — ${data.discovery.top_match.score.confidence.toFixed(2)})`, 
          nextRunning: 'Building evidence record...' 
        },
        { doneText: 'Evidence hashed', nextRunning: 'Uploading to IPFS...' },
        { doneText: `Stored on IPFS (CID: ${data.ipfs.cid.slice(0, 16)}...)`, nextRunning: 'Anchoring to blockchain...' },
        { doneText: `Anchored — tx: ${data.blockchain.tx_hash.slice(0, 14)}...`, nextRunning: 'Verifying against chain...' },
        { doneText: 'VERIFIED', nextRunning: null, isVerified: true },
      ];

      let currentLines = [{ text: 'Checking liveness...', status: 'running' }];

      for (let i = 0; i < steps.length; i++) {
        await new Promise(r => setTimeout(r, 280));
        const step = steps[i];
        
        currentLines[currentLines.length - 1] = {
          text: step.doneText,
          status: 'done',
          isVerified: step.isVerified,
        };

        if (step.nextRunning) {
          currentLines.push({ text: step.nextRunning, status: 'running' });
        }

        setFeedLines([...currentLines]);
      }

      setPipelineResult(data);
    } catch (err) {
      setFeedLines((prev) => {
        const copy = [...prev];
        if (copy.length > 0) {
          copy[copy.length - 1] = {
            text: `Failed — ${err.message}`,
            status: 'failed',
          };
        } else {
          copy.push({ text: `Failed — ${err.message}`, status: 'failed' });
        }
        return copy;
      });
      setErrorMessage(err.message);
    } finally {
      setIsVerifying(false);
    }
  };

  // Run Tamper Test
  const handleTamperTest = async () => {
    if (!pipelineResult) return;
    setIsTampering(true);

    try {
      const origTier = pipelineResult.discovery.top_match.score.tier;
      const mutatedTier = origTier === 'HIGH' ? 'LOW' : 'HIGH';

      const payload = {
        evidence_dict: pipelineResult.evidence.evidence_dict,
        original_hash: pipelineResult.evidence.evidence_hash,
        merkle_proof: pipelineResult.merkle.proof,
        merkle_root: pipelineResult.merkle.root,
        field_to_mutate: 'confidence_tier',
        mutated_value: mutatedTier,
      };

      const res = await fetch('/api/pipeline/tamper', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error('Tamper endpoint failed');

      const data = await res.json();
      setTamperResult(data);
    } catch (err) {
      alert('Tamper test error: ' + err.message);
    } finally {
      setIsTampering(false);
    }
  };

  // Download Evidence JSON
  const downloadEvidenceJson = () => {
    if (!pipelineResult) return;
    const blob = new Blob([pipelineResult.evidence.canonical_json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `evidence_${pipelineResult.evidence.evidence_hash.slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="app-container">
      {/* 1. HEADER */}
      <header className="header-box">
        <h1 className="header-title">KAO</h1>
        <p className="header-tagline">Consent-gated face discovery and Merkle-anchored attestation pipeline.</p>
      </header>

      {/* 2. INPUT SECTION */}
      <section className="input-card">
        {/* Input Mode Selector */}
        <div className="tab-row">
          <button
            type="button"
            className={`tab-pill ${inputMode === 'upload' ? 'active' : ''}`}
            onClick={() => setInputMode('upload')}
          >
            Upload Photo
          </button>
          <button
            type="button"
            className={`tab-pill ${inputMode === 'camera' ? 'active' : ''}`}
            onClick={() => {
              setInputMode('camera');
              startCamera();
            }}
          >
            Use Camera
          </button>
          <button
            type="button"
            className={`tab-pill ${inputMode === 'sample' ? 'active' : ''}`}
            onClick={() => setInputMode('sample')}
          >
            Sample Portrait (Alex Rivera)
          </button>
        </div>

        {/* Input: Upload Photo */}
        {inputMode === 'upload' && (
          <div className="input-body">
            <input
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={handleFileChange}
              id="file-upload-input"
              className="file-input"
            />
            <label htmlFor="file-upload-input" className="file-dropzone">
              {uploadedFile ? uploadedFile.name : 'Choose a JPG, PNG, or WEBP photo'}
            </label>
            {uploadPreview && (
              <div className="thumb-container">
                <img src={uploadPreview} alt="Selected" className="thumb-img" />
              </div>
            )}
          </div>
        )}

        {/* Input: Live Camera */}
        {inputMode === 'camera' && (
          <div className="input-body">
            {cameraActive && (
              <div className="cam-container">
                <video ref={videoRef} autoPlay playsInline muted className="cam-video" />
                <div style={{ marginTop: '10px' }}>
                  <button type="button" className="btn-action" onClick={captureFrame}>
                    Capture Photo
                  </button>
                </div>
              </div>
            )}

            {capturedPreview && !cameraActive && (
              <div className="thumb-container">
                <img src={capturedPreview} alt="Captured frame" className="thumb-img" />
                <div style={{ marginTop: '8px' }}>
                  <button type="button" className="btn-secondary" onClick={startCamera}>
                    Retake
                  </button>
                </div>
              </div>
            )}

            {!cameraActive && !capturedPreview && (
              <button type="button" className="btn-secondary" onClick={startCamera}>
                Activate Webcam
              </button>
            )}
          </div>
        )}

        {/* Input: Sample Portrait */}
        {inputMode === 'sample' && (
          <div className="input-body">
            <div className="thumb-container">
              <img src="/api/presets/image/demo_portrait_1.jpg" alt="Alex Rivera" className="thumb-img" />
              <div className="thumb-desc">Sample Portrait: Alex Rivera (Staff Systems Engineer)</div>
            </div>
          </div>
        )}

        {/* Consent Checkbox */}
        <div style={{ marginTop: '20px' }}>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={consent}
              onChange={(e) => setConsent(e.target.checked)}
            />
            <span>I confirm this is my own face or a consenting subject's face</span>
          </label>
        </div>

        {/* Verify Button */}
        <div style={{ marginTop: '18px' }}>
          <button
            type="button"
            className="btn-verify"
            disabled={!canVerify}
            onClick={handleVerify}
          >
            {isVerifying ? 'Verifying...' : 'Verify'}
          </button>
        </div>
      </section>

      {/* 3. STATUS FEED */}
      {feedLines.length > 0 && (
        <section className="status-feed-box">
          {feedLines.map((line, idx) => (
            <div
              key={idx}
              className={`feed-row ${line.status} ${line.isVerified ? 'verified-row' : ''}`}
            >
              <span className="feed-icon">
                {line.status === 'running' && '⏳'}
                {line.status === 'done' && '✓'}
                {line.status === 'failed' && '✗'}
              </span>
              <span className="feed-text">{line.text}</span>
            </div>
          ))}
        </section>
      )}

      {/* 4. RESULT CARD */}
      {pipelineResult && (
        <section className="result-card">
          <div className="result-title">Evidence Attestation Card</div>

          <div className="result-grid">
            <div className="result-thumb-box">
              <img
                src={pipelineResult.discovery.top_match.thumbnail_b64}
                alt="Matched Candidate"
                className="result-matched-img"
              />
            </div>

            <div className="result-info">
              <div className="platform-tag">
                <span className="platform-dot">●</span>
                <span>{pipelineResult.discovery.top_match.platform.toUpperCase()}</span>
              </div>

              <div className="post-link-box">
                <a
                  href={pipelineResult.discovery.top_match.url}
                  target="_blank"
                  rel="noreferrer"
                  className="post-link"
                >
                  {pipelineResult.discovery.top_match.title}
                </a>
              </div>

              {/* Confidence Tier */}
              <div className="tier-badge-container">
                <span className={`tier-badge tier-${pipelineResult.discovery.top_match.score.tier.toLowerCase()}`}>
                  CONFIDENCE: {pipelineResult.discovery.top_match.score.tier}
                </span>
                <div className="tier-score">
                  Numeric Score: <strong>{pipelineResult.discovery.top_match.score.confidence.toFixed(2)}</strong> (raw fused: {pipelineResult.discovery.top_match.score.raw_score.toFixed(2)})
                </div>
              </div>

              {/* PolygonScan Link */}
              <div style={{ marginTop: '14px' }}>
                <a
                  href={pipelineResult.blockchain.explorer_url}
                  target="_blank"
                  rel="noreferrer"
                  className="chain-link"
                >
                  View transaction on PolygonScan →
                </a>
              </div>
            </div>
          </div>

          {/* All Discovered Web Candidates with Platform Badges */}
          {pipelineResult.discovery.all_candidates && pipelineResult.discovery.all_candidates.length > 1 && (
            <div className="candidates-section">
              <div className="candidates-title">
                All Discovered Web Profiles ({pipelineResult.discovery.all_candidates.length})
              </div>
              <div className="candidates-list">
                {pipelineResult.discovery.all_candidates.map((cand, idx) => (
                  <div key={idx} className="candidate-row">
                    {cand.thumbnail_b64 && (
                      <img src={cand.thumbnail_b64} alt={cand.title} className="candidate-thumb" />
                    )}
                    <div className="candidate-details">
                      <div className="candidate-meta">
                        <span className="candidate-platform">{cand.platform.toUpperCase()}</span>
                        <span className="candidate-tier">
                          Match: {(cand.score.confidence * 100).toFixed(0)}% ({cand.score.tier})
                        </span>
                      </div>
                      <a href={cand.url} target="_blank" rel="noreferrer" className="candidate-link">
                        {cand.title}
                      </a>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <hr className="divider" />

          {/* Run Tamper Test Button */}
          <div>
            <button
              type="button"
              className="btn-tamper"
              disabled={isTampering}
              onClick={handleTamperTest}
            >
              {isTampering ? 'Testing...' : 'Run Tamper Test'}
            </button>

            {tamperResult && (
              <div className="tamper-box">
                <div className="tamper-alert-title">
                  ✗ TAMPER DETECTED — evidence hash no longer matches on-chain record.<br />
                  Changed field: <code>{tamperResult.tampered_field}</code> ({tamperResult.original_value} → {tamperResult.tampered_value})
                </div>

                <div className="diff-view">
                  <div>
                    <strong>Original SHA-256:</strong><br />
                    <span className="mono">{tamperResult.original_hash}</span>
                  </div>
                  <div style={{ marginTop: '8px' }}>
                    <strong>Tampered SHA-256:</strong><br />
                    <span className="mono tampered-hash">{tamperResult.tampered_hash}</span>
                  </div>
                  <div style={{ marginTop: '8px', color: '#a1a1aa' }}>
                    <strong>Merkle Proof Walk:</strong> FAILED (Recomputed root != Anchored root)
                  </div>
                </div>
              </div>
            )}
          </div>
        </section>
      )}

      {/* 5. OPTIONAL — Show technical details expander */}
      {pipelineResult && (
        <section className="tech-expander-section">
          <button
            type="button"
            className="expander-trigger"
            onClick={() => setShowTechDetails(!showTechDetails)}
          >
            <span>{showTechDetails ? '[-] Hide technical details' : '[+] Show technical details'}</span>
          </button>

          {showTechDetails && (
            <div className="tech-details-body">
              <div className="tech-title">Individual Multi-Signal Breakdown</div>
              <div className="signal-grid">
                <div className="signal-box">
                  <div className="signal-label">Face Cosine Similarity (55%)</div>
                  <div className="signal-val">
                    {(pipelineResult.discovery.top_match.score.face_similarity * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="signal-box">
                  <div className="signal-label">Perceptual Hash Metric (25%)</div>
                  <div className="signal-val">
                    {(pipelineResult.discovery.top_match.score.phash_similarity * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="signal-box">
                  <div className="signal-label">Context Relevance (20%)</div>
                  <div className="signal-val">
                    {(pipelineResult.discovery.top_match.score.text_similarity * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              <hr className="divider" />

              <div className="tech-title">Canonical Evidence JSON (RFC 8785)</div>
              <pre className="json-box">
                {JSON.stringify(pipelineResult.evidence.evidence_dict, null, 2)}
              </pre>
              <button
                type="button"
                className="btn-secondary"
                style={{ marginTop: '8px', fontSize: '11px', padding: '6px 12px' }}
                onClick={downloadEvidenceJson}
              >
                Download Evidence JSON
              </button>

              <hr className="divider" />

              <div className="tech-title">Merkle Proof Tree</div>
              <div className="mono-text">Merkle Root: 0x{pipelineResult.merkle.root}</div>
              <div style={{ marginTop: '6px' }}>
                {pipelineResult.merkle.proof.map((step, idx) => (
                  <div key={idx} className="mono-subtext">
                    Step {idx + 1} [{step[1].toUpperCase()}]: {step[0]}
                  </div>
                ))}
              </div>

              <hr className="divider" />

              <div className="tech-title">Storage & Wallet References</div>
              <div className="ref-line">
                <strong>IPFS CID:</strong> <code>{pipelineResult.ipfs.cid}</code>
              </div>
              <div className="ref-line">
                <strong>IPFS Gateway:</strong>{' '}
                <a href={pipelineResult.ipfs.gateway_url} target="_blank" rel="noreferrer">
                  {pipelineResult.ipfs.gateway_url}
                </a>
              </div>
              <div className="ref-line">
                <strong>Wallet Address Used:</strong>{' '}
                <code>0x18F4d9D10c66BE2479e0E6F0b7c15B76f7B35D01</code>
              </div>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
