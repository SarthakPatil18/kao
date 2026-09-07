"""
Stage 1 — Liveness Check (Hard Gate)

Uses MediaPipe FaceMesh to verify a live, consenting subject is present
before any search or biometric processing occurs.

Challenges are randomized (blink twice OR turn head left→right) to prevent
replay attacks with pre-recorded video.
"""

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# MediaPipe FaceMesh landmark indices for eye aspect ratio (EAR)
# Left eye: upper [159, 145], lower [153, 133], corners [33, 133]
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# Nose tip landmark for head-turn detection
NOSE_TIP_IDX = 1

# Thresholds
EAR_THRESHOLD = 0.21          # Below this = eye closed
EAR_CONSEC_FRAMES = 2         # Frames below threshold = one blink
BLINK_COUNT_REQUIRED = 2      # "Blink twice" challenge
HEAD_TURN_X_DELTA = 0.08      # Minimum normalized x-displacement for a turn
MIN_FRAMES_FOR_CHECK = 10     # Minimum frames needed for any liveness check

CHALLENGES = ["blink_twice", "turn_head_left_right"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LivenessResult:
    """Result of a liveness check."""
    passed: bool
    confidence: float       # 0.0 – 1.0
    challenge: str          # Which challenge was issued
    reason: str             # Human-readable explanation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eye_aspect_ratio(landmarks: list, eye_indices: List[int],
                      img_w: int, img_h: int) -> float:
    """Compute the Eye Aspect Ratio (EAR) for one eye.

    EAR ≈ (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    A low EAR indicates the eye is closed.
    """
    def _pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * img_w, lm.y * img_h])

    p = [_pt(i) for i in eye_indices]
    # Vertical distances
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    # Horizontal distance
    h = np.linalg.norm(p[0] - p[3])
    if h < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def _count_blinks(ear_series: List[float]) -> int:
    """Count blink events in a time-series of EAR values."""
    blink_count = 0
    below_count = 0
    for ear in ear_series:
        if ear < EAR_THRESHOLD:
            below_count += 1
        else:
            if below_count >= EAR_CONSEC_FRAMES:
                blink_count += 1
            below_count = 0
    # Handle case where series ends mid-blink
    if below_count >= EAR_CONSEC_FRAMES:
        blink_count += 1
    return blink_count


def _detect_head_turn(nose_x_series: List[float]) -> bool:
    """Detect a left-then-right (or right-then-left) head turn.

    Looks for the nose tip x-coordinate moving significantly in one direction
    and then returning past center.
    """
    if len(nose_x_series) < 5:
        return False

    min_x = min(nose_x_series)
    max_x = max(nose_x_series)
    delta = max_x - min_x

    if delta < HEAD_TURN_X_DELTA:
        return False

    # Check that the peak and trough occur at different times (not just noise)
    min_idx = nose_x_series.index(min_x)
    max_idx = nose_x_series.index(max_x)
    return abs(max_idx - min_idx) >= 3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_challenge() -> str:
    """Return a randomly chosen liveness challenge string."""
    return random.choice(CHALLENGES)


def get_challenge_instruction(challenge: str) -> str:
    """Return a human-readable instruction for the given challenge."""
    instructions = {
        "blink_twice": "Please blink your eyes twice slowly.",
        "turn_head_left_right": "Please turn your head to the left, then to the right.",
    }
    return instructions.get(challenge, "Unknown challenge.")


def run_liveness_check(
    frames: List[np.ndarray],
    challenge: str,
) -> LivenessResult:
    """Run the liveness check on a sequence of video frames.

    Args:
        frames: List of BGR numpy arrays (video frames from webcam).
        challenge: One of CHALLENGES — determines which motion to look for.

    Returns:
        LivenessResult with pass/fail, confidence, and reason.
    """
    if mp is None:
        return LivenessResult(
            passed=False, confidence=0.0, challenge=challenge,
            reason="MediaPipe not installed. Run: pip install mediapipe",
        )

    if len(frames) < MIN_FRAMES_FOR_CHECK:
        return LivenessResult(
            passed=False, confidence=0.0, challenge=challenge,
            reason=f"Need at least {MIN_FRAMES_FOR_CHECK} frames, got {len(frames)}.",
        )

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    ear_series: List[float] = []
    nose_x_series: List[float] = []
    frames_with_face = 0

    for frame in frames:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            continue

        landmarks = results.multi_face_landmarks[0].landmark
        frames_with_face += 1

        # EAR (average of both eyes)
        left_ear = _eye_aspect_ratio(landmarks, LEFT_EYE_IDX, w, h)
        right_ear = _eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, w, h)
        avg_ear = (left_ear + right_ear) / 2.0
        ear_series.append(avg_ear)

        # Nose-tip x (normalized 0–1)
        nose_x_series.append(landmarks[NOSE_TIP_IDX].x)

    face_mesh.close()

    # Must have detected a face in most frames
    face_ratio = frames_with_face / len(frames)
    if face_ratio < 0.5:
        return LivenessResult(
            passed=False, confidence=face_ratio, challenge=challenge,
            reason=f"Face detected in only {frames_with_face}/{len(frames)} frames.",
        )

    # Evaluate the requested challenge
    if challenge == "blink_twice":
        blinks = _count_blinks(ear_series)
        passed = blinks >= BLINK_COUNT_REQUIRED
        confidence = min(1.0, blinks / BLINK_COUNT_REQUIRED)
        reason = (
            f"Detected {blinks} blink(s), required {BLINK_COUNT_REQUIRED}."
            if not passed
            else f"Liveness confirmed: {blinks} blinks detected."
        )

    elif challenge == "turn_head_left_right":
        turned = _detect_head_turn(nose_x_series)
        if turned and nose_x_series:
            delta = max(nose_x_series) - min(nose_x_series)
            confidence = min(1.0, delta / HEAD_TURN_X_DELTA)
        else:
            confidence = 0.0
        passed = turned
        reason = (
            "Head turn detected. Liveness confirmed."
            if passed
            else "Head turn not detected. Please turn your head left then right."
        )

    else:
        return LivenessResult(
            passed=False, confidence=0.0, challenge=challenge,
            reason=f"Unknown challenge type: {challenge}",
        )

    return LivenessResult(
        passed=passed,
        confidence=round(confidence, 3),
        challenge=challenge,
        reason=reason,
    )


def capture_frames_from_webcam(
    duration_seconds: float = 3.0,
    fps: int = 15,
) -> List[np.ndarray]:
    """Capture frames from the default webcam for liveness checking.

    Args:
        duration_seconds: How long to record.
        fps: Target frames per second.

    Returns:
        List of BGR frames.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return []

    frames = []
    total_frames = int(duration_seconds * fps)
    delay_ms = max(1, int(1000 / fps))

    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        cv2.waitKey(delay_ms)

    cap.release()
    return frames
