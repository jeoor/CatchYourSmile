"""Gesture recognition from MediaPipe hand landmarks."""

from collections import deque
import numpy as np

from config import V_SIGN_DISTANCE_THRESHOLD, OK_DISTANCE_THRESHOLD


def check_gestures(landmarks):
    """Detect gesture from hand landmarks.

    Returns one of: 'FIST', 'V', 'OK', 'OPEN', or None.
    """
    idx_up = landmarks[8].y < landmarks[6].y
    mid_up = landmarks[12].y < landmarks[10].y
    rng_up = landmarks[16].y < landmarks[14].y
    pnk_up = landmarks[20].y < landmarks[18].y

    if not (idx_up or mid_up or rng_up or pnk_up):
        return "FIST"

    if idx_up and mid_up and not rng_up and not pnk_up:
        dist = np.hypot(landmarks[8].x - landmarks[12].x,
                        landmarks[8].y - landmarks[12].y)
        if dist > V_SIGN_DISTANCE_THRESHOLD:
            return "V"
        return None

    if np.hypot(landmarks[4].x - landmarks[8].x,
                landmarks[4].y - landmarks[8].y) < OK_DISTANCE_THRESHOLD:
        return "OK"

    if idx_up and mid_up and rng_up and pnk_up:
        return "OPEN"

    return None


class GestureStabilizer:
    """Debounce gesture detection using consensus over a sliding window."""

    def __init__(self, buffer_size=5, min_consensus=3):
        self._buffer = deque(maxlen=buffer_size)
        self._min_consensus = min_consensus

    def update(self, raw_gesture):
        """Feed a raw gesture, return stable gesture or None."""
        self._buffer.append(raw_gesture)
        if len(self._buffer) < 3:
            return None

        valid = [g for g in self._buffer if g is not None]
        if not valid:
            return None

        most_common = max(set(valid), key=valid.count)
        if valid.count(most_common) >= self._min_consensus:
            return most_common
        return None

    def reset(self):
        self._buffer.clear()
