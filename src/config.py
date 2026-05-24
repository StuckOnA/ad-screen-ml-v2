# =============================================================================
# config.py
# =============================================================================

# --- Video ---
VIDEO_PATH   = "videos/test5.mp4"
DISPLAY_SIZE = (1280, 720)
FRAME_SKIP   = 2

# --- Models ---
YOLO_MODEL      = "yolov8n.pt"       # back to standard detection model
INSIGHTFACE_DET = (640, 640)

# --- Detection ---
MIN_DETECTION_CONF = 0.4

# --- Stability thresholds ---
STABLE_FRAMES_REQUIRED    = 15
STABLE_CONF_THRESHOLD     = 0.55

PRECISION_FRAMES_REQUIRED = 40
PRECISION_CONF_THRESHOLD  = 0.75
PRECISION_BBOX_AREA       = 45000

# --- Facing away (miss-based, no pose model) ---
FACING_AWAY_MISS_THRESHOLD         = 3   # previously labeled person
FACING_AWAY_INITIAL_THRESHOLD      = 2   # never labeled person
FACING_AWAY_RECHECK_INTERVAL       = 4.0 # seconds before retrying a facing-away person

# --- Reanalysis buckets ---
REANALYZE_BUCKETS = {
    "high":   15.0,
    "medium":  7.0,
    "low":     2.0,
}
REANALYZE_NONE = 1.0

# --- Agreement thresholds (for reanalysis frequency) ---
# Based on consistency of gender history, not detection confidence
AGREEMENT_HIGH_THRESHOLD  = 0.80   # 4/5+ agree → slow reanalysis
AGREEMENT_LOW_THRESHOLD   = 0.60   # 3/5+ agree → medium reanalysis

# --- Stale ID timeout ---
STALE_TIMEOUT = 5.0

# --- Ad cooldown ---
AD_COOLDOWN = 2.0

# --- Ads ---
AD_PATHS = {
    "ad1": "ads/ad1.jpg",
    "ad2": "ads/ad2.jpg",
}
