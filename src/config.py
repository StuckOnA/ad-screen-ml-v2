# =============================================================================
# config.py — all tunable constants in one place
# =============================================================================

# --- Video ---
VIDEO_PATH   = "videos/test5.mp4"
DISPLAY_SIZE = (1280, 720)
FRAME_SKIP   = 2

# --- Models ---
YOLO_MODEL      = "yolov8n.pt"
INSIGHTFACE_DET = (320, 320)

# --- Detection ---
MIN_DETECTION_CONF = 0.4

# --- Stability thresholds ---
STABLE_FRAMES_REQUIRED    = 15
STABLE_CONF_THRESHOLD     = 0.55

PRECISION_FRAMES_REQUIRED = 40
PRECISION_CONF_THRESHOLD  = 0.75
PRECISION_BBOX_AREA       = 45000

# --- Facing away ---
FACING_AWAY_MISS_THRESHOLD         = 3   # misses before marking away (previously labeled)
FACING_AWAY_INITIAL_THRESHOLD      = 1   # misses before marking away (never labeled)

# --- Reanalysis buckets (seconds between retries) ---
REANALYZE_BUCKETS = {
    "high":   15.0,   # confident result
    "medium":  7.0,
    "low":     2.0,   # weak result — retry often
}
REANALYZE_NONE = 1.0  # never analyzed yet — try fast

# --- Stale ID timeout (seconds) ---
STALE_TIMEOUT = 5.0

# --- Ad cooldown (seconds between switches) ---
AD_COOLDOWN = 2.0

# --- Ads ---
AD_PATHS = {
    "ad1": "ads/ad1.jpg",
    "ad2": "ads/ad2.jpg",
}