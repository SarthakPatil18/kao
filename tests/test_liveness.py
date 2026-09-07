"""
Tests for liveness check logic.

Tests the challenge detection functions with synthetic data
(EAR series for blinks, nose-x series for head turns).
"""

import numpy as np
import pytest
from pipeline.liveness import (
    _count_blinks,
    _detect_head_turn,
    generate_challenge,
    get_challenge_instruction,
    run_liveness_check,
    CHALLENGES,
    EAR_THRESHOLD,
    BLINK_COUNT_REQUIRED,
    HEAD_TURN_X_DELTA,
)


# ---------------------------------------------------------------------------
# Blink detection tests
# ---------------------------------------------------------------------------

class TestBlinkDetection:
    """Eye aspect ratio blink counting."""

    def test_no_blinks(self):
        """Constant open EAR → 0 blinks."""
        ear_series = [0.30] * 30  # Always open
        assert _count_blinks(ear_series) == 0

    def test_one_blink(self):
        """One dip below threshold → 1 blink."""
        ear_series = [0.30] * 10 + [0.15, 0.14, 0.15] + [0.30] * 10
        assert _count_blinks(ear_series) == 1

    def test_two_blinks(self):
        """Two separate dips → 2 blinks."""
        ear_series = (
            [0.30] * 5
            + [0.15, 0.14, 0.15]  # blink 1
            + [0.30] * 5
            + [0.14, 0.13, 0.14]  # blink 2
            + [0.30] * 5
        )
        assert _count_blinks(ear_series) == 2

    def test_three_blinks(self):
        """Three blinks detected."""
        ear_series = []
        for _ in range(3):
            ear_series.extend([0.30] * 5 + [0.15, 0.14, 0.15])
        ear_series.extend([0.30] * 5)
        assert _count_blinks(ear_series) == 3

    def test_single_frame_dip_ignored(self):
        """Single frame dip (< EAR_CONSEC_FRAMES) not counted as blink."""
        ear_series = [0.30] * 10 + [0.15] + [0.30] * 10
        assert _count_blinks(ear_series) == 0

    def test_sustained_closure_counts_as_one(self):
        """Eyes closed for many frames counts as one blink."""
        ear_series = [0.30] * 5 + [0.15] * 20 + [0.30] * 5
        assert _count_blinks(ear_series) == 1


# ---------------------------------------------------------------------------
# Head turn detection tests
# ---------------------------------------------------------------------------

class TestHeadTurnDetection:
    """Nose-tip x-position head turn detection."""

    def test_no_movement(self):
        """Stationary head → no turn detected."""
        nose_x = [0.50] * 20
        assert _detect_head_turn(nose_x) is False

    def test_clear_left_right_turn(self):
        """Clear left→right head turn detected."""
        # Head moves from center (0.5) to left (0.3) to right (0.7)
        nose_x = (
            [0.50] * 3
            + [0.45, 0.40, 0.35, 0.30]  # Turn left
            + [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]  # Turn right
            + [0.65, 0.60, 0.55, 0.50]  # Return center
        )
        assert _detect_head_turn(nose_x) is True

    def test_small_movement_rejected(self):
        """Small head movement below threshold not detected."""
        nose_x = [0.50 + 0.01 * (i % 3 - 1) for i in range(20)]
        assert _detect_head_turn(nose_x) is False

    def test_too_few_frames(self):
        """Less than 5 frames → no detection possible."""
        assert _detect_head_turn([0.30, 0.70]) is False
        assert _detect_head_turn([]) is False


# ---------------------------------------------------------------------------
# Challenge generation tests
# ---------------------------------------------------------------------------

class TestChallengeGeneration:
    """Challenge selection."""

    def test_valid_challenge(self):
        """Generated challenge is from the known set."""
        for _ in range(20):
            c = generate_challenge()
            assert c in CHALLENGES

    def test_instruction_exists(self):
        """Every challenge has a human-readable instruction."""
        for challenge in CHALLENGES:
            instruction = get_challenge_instruction(challenge)
            assert len(instruction) > 0
            assert instruction != "Unknown challenge."


# ---------------------------------------------------------------------------
# Integration test (with synthetic frames)
# ---------------------------------------------------------------------------

class TestLivenessIntegration:
    """Integration tests using too-few-frames edge case."""

    def test_too_few_frames_rejected(self):
        """Less than MIN_FRAMES frames → liveness fails."""
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)] * 3
        result = run_liveness_check(frames, "blink_twice")
        assert result.passed is False
        # Accepts either "not enough frames" or "mediapipe not installed" as valid failure
        reason = result.reason.lower()
        assert any(kw in reason for kw in ("frames", "need", "not installed"))

    def test_empty_frames_rejected(self):
        """Empty frame list → liveness fails."""
        result = run_liveness_check([], "blink_twice")
        assert result.passed is False
