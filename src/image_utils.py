"""Image processing utilities — drawing, text, watermarks."""

import os
import cv2
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont


def find_chinese_font(font_paths):
    """Return first available Chinese font path, or None."""
    for path in font_paths:
        if os.path.exists(path):
            print(f"找到字体: {path}")
            return path
    return None


def draw_rounded_rect(img, pt1, pt2, color, radius=10, thickness=-1):
    """Draw a rounded rectangle on an image."""
    x1, y1 = pt1
    x2, y2 = pt2

    if thickness == -1:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for cx, cy in [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]:
            cv2.circle(img, (cx, cy), radius, color, -1)
    else:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        angles = [(180, 270), (270, 360), (90, 180), (0, 90)]
        corners = [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                   (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]
        for (cx, cy), (sa, ea) in zip(corners, angles):
            cv2.ellipse(img, (cx, cy), (radius, radius), 0, sa, ea, color, thickness)


def create_gradient_background(width, height, dark_color):
    """Create a vertical dark gradient background image."""
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        factor = 1.0 - (y / height) * 0.3
        gradient[y, :] = (np.array(dark_color) * factor).astype(np.uint8)
    return gradient


def add_timestamp_to_image(image):
    """Add header text and timestamp watermark to a photo.

    Returns (annotated_image, filename_prefix).
    """
    h, w = image.shape[:2]
    now = datetime.now()
    filename = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    display_time = now.strftime("%Y-%m-%d %H:%M:%S")

    image = _add_top_text(image, w, h)
    image = _add_timestamp_watermark(image, w, h, display_time)
    return image, filename


def _add_top_text(image, width, height):
    """Add 'Keep smile everyday!' header with semi-transparent background."""
    text = "Keep smile everyday!"
    font_scale = 1.2
    thickness = 3
    padding = 20

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx = (width - tw) // 2
    ty = th + padding

    overlay = image.copy()
    cv2.rectangle(overlay,
                  (tx - padding // 2, ty - th - padding // 2),
                  (tx + tw + padding // 2, ty + padding // 2),
                  (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
    cv2.putText(image, text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return image


def _add_timestamp_watermark(image, width, height, timestamp):
    """Add timestamp at bottom-right with semi-transparent background."""
    font_scale = 0.6
    thickness = 2
    padding = 10

    (tw, th), _ = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    px = width - tw - padding
    py = height - padding

    overlay = image.copy()
    cv2.rectangle(overlay,
                  (px - padding // 2, py - th - padding // 2),
                  (px + tw + padding // 2, py + padding // 2),
                  (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)
    cv2.putText(image, timestamp, (px, py),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return image
