"""Configuration constants for CatchYourSmile."""

# Color theme
THEME = {
    'primary': (0, 180, 255),
    'secondary': (0, 255, 200),
    'success': (0, 255, 127),
    'danger': (255, 80, 80),
    'warning': (255, 200, 50),
    'dark': (15, 20, 30),
    'light': (240, 245, 255),
    'gray': (60, 70, 85),
    'accent': (180, 100, 255),
}

# Chinese font search paths (Windows)
FONT_PATHS = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/simkai.ttf",
    "C:/Windows/Fonts/msjh.ttc",
]

# Photo save directory
SAVE_DIR = "captured_photos"

# MediaPipe model settings
HAND_MAX_NUM = 1
HAND_DETECTION_CONFIDENCE = 0.8
HAND_TRACKING_CONFIDENCE = 0.8
FACE_MAX_NUM = 1
FACE_DETECTION_CONFIDENCE = 0.5
FACE_TRACKING_CONFIDENCE = 0.5

# Timing thresholds (seconds)
SMILE_COOLDOWN = 5.0
SMILE_FRAMES_NEEDED = 12
UNLOCK_COOLDOWN = 5.0
EXIT_THRESHOLD = 5.0
V_PHOTO_COOLDOWN = 1.0
FLASH_DURATION = 0.5
STATUS_DURATION = 3.0
PREVIEW_DISPLAY_DURATION = 5.0
COUNTDOWN_DURATION = 3
NO_SMILE_RESET_DELAY = 0.5

# Gesture detection
GESTURE_BUFFER_SIZE = 5
GESTURE_MIN_CONSENSUS = 3
V_SIGN_DISTANCE_THRESHOLD = 0.03
OK_DISTANCE_THRESHOLD = 0.04

# Camera defaults
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
CANVAS_TOP_MARGIN = 80
CANVAS_SIDE_MARGIN = 40
BORDER_SIZE = 2
