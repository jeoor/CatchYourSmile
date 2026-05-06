"""Smile detection using MediaPipe face mesh mouth keypoints."""

import math

# MediaPipe face mesh mouth landmark indices
_MOUTH_IDX = {
    'left_corner': 61, 'right_corner': 291,
    'top_center': 0, 'bottom_center': 17,
    'inner_left': 78, 'inner_right': 308,
    'inner_top': 13, 'inner_bottom': 14,
}


def _get_mouth_landmarks(face_landmarks, img_width, img_height):
    """Extract mouth keypoints in pixel coordinates."""
    points = {}
    for name, idx in _MOUTH_IDX.items():
        lm = face_landmarks.landmark[idx]
        points[name] = (int(lm.x * img_width), int(lm.y * img_height))

    pts = points
    points_tuple = (
        pts['left_corner'][0], pts['left_corner'][1],
        pts['right_corner'][0], pts['right_corner'][1],
        pts['top_center'][0], pts['top_center'][1],
        pts['bottom_center'][0], pts['bottom_center'][1],
    )
    return {'points': points_tuple, 'coords': points}


def _dist(p1, p2):
    """Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _calculate_mouth_features(landmarks):
    """Calculate mouth shape measurements."""
    c = landmarks['coords']

    outer_width = _dist(c['right_corner'], c['left_corner'])
    outer_height = _dist(c['bottom_center'], c['top_center'])
    inner_width = _dist(c['inner_right'], c['inner_left'])
    inner_height = _dist(c['inner_bottom'], c['inner_top'])

    face_width = abs(c['right_corner'][0] * 1.1 - c['left_corner'][0] * 0.9)

    return {
        'outer_width': max(outer_width, 1),
        'outer_height': max(outer_height, 1),
        'inner_width': max(inner_width, 1),
        'inner_height': max(inner_height, 1),
        'face_width': max(face_width, 1),
    }


def _calculate_smile_score(features):
    """Compute weighted smile score from mouth features."""
    outer_ratio = features['outer_width'] / features['outer_height']
    inner_ratio = features['inner_width'] / features['inner_height']
    mouth_to_face = features['outer_width'] / features['face_width']

    return outer_ratio * 0.4 + inner_ratio * 0.3 + mouth_to_face * 10 * 0.3


def detect_smile(face_landmarks, img_width, img_height,
                 smile_threshold=0.5, smile_sensitivity=0.5):
    """Detect whether face is smiling.

    Args:
        face_landmarks: MediaPipe face mesh landmarks.
        img_width, img_height: Image dimensions in pixels.
        smile_threshold: Base threshold for smile score (default 0.5).
        smile_sensitivity: User-adjusted sensitivity 0.0-1.0 (default 0.5).

    Returns:
        (is_smiling, smile_score, mouth_points)
        mouth_points: (lx, ly, rx, ry, tx, ty, bx, by) in pixel coords.
    """
    try:
        landmarks = _get_mouth_landmarks(face_landmarks, img_width, img_height)
        features = _calculate_mouth_features(landmarks)
        smile_score = _calculate_smile_score(features)
        adjusted_threshold = smile_threshold * (1.5 - smile_sensitivity)
        is_smiling = (smile_score > adjusted_threshold and
                      features['inner_height'] > 5)
        return is_smiling, smile_score, landmarks['points']
    except Exception:
        return False, 0, (0, 0, 0, 0, 0, 0, 0, 0)
