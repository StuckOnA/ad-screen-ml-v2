# =============================================================================
# main.py — entry point
#
# Orchestrates:
#   vision/detector.py    → YOLO detections
#   vision/identity.py    → per-person memory
#   vision/workers.py     → IF + DF background analysis
#
# Decision engine and display modules to be added next.
# =============================================================================

import cv2
import numpy as np
import time
import torch
from insightface.app import FaceAnalysis

from config import (
    VIDEO_PATH,
    DISPLAY_SIZE,
    FRAME_SKIP,
    INSIGHTFACE_DET,
    REANALYZE_BUCKETS,
    AD_COOLDOWN,
    AD_PATHS,
    PRECISION_FRAMES_REQUIRED,
    PRECISION_CONF_THRESHOLD,
)
from vision.detector  import Detector
from vision.identity  import (
    identity_memory,
    memory_lock,
    new_entry,
    get_snapshot,
    prune_stale_identities
)
from vision.workers import (
    setup           as workers_setup,
    start_workers,
    enqueue_insightface,
    enqueue_deepface,
)

# ---------------------------------------------------------------------------
# Local Helper Functions (Restored for Tracking Logic)
# ---------------------------------------------------------------------------
def get_entry_snapshot(person_id):
    """Safely retrieves a copy of a single person's memory."""
    with memory_lock:
        mem = identity_memory.get(person_id)
        return mem.copy() if mem else None

def update_entry(mem, conf, now):
    """Updates spatial tracking stats for an existing ID."""
    mem["last_seen"] = now
    mem["frames_seen"] = mem.get("frames_seen", 0) + 1
    # Moving average for confidence smoothing
    mem["avg_conf"] = (mem.get("avg_conf", conf) * 0.9) + (conf * 0.1)

def is_stable(snap):
    """Determines if a tracked person has been around long enough to be analyzed."""
    return snap.get("frames_seen", 0) >= 10 and snap.get("avg_conf", 0) >= 0.55

def get_analysis_tier(snap, bbox_area):
    """Determines which AI worker should process this person."""
    if not is_stable(snap):
        return "noisy"
    if snap.get("frames_seen", 0) >= PRECISION_FRAMES_REQUIRED and snap.get("avg_conf", 0) >= PRECISION_CONF_THRESHOLD:
        return "precision"
    return "standard"

def get_reanalyze_interval(snap):
    """Fetches the cooldown time before re-sending a face to the AI."""
    conf = snap.get("avg_conf", 0)
    if conf >= 0.80:
        return REANALYZE_BUCKETS.get("high", 5.0)
    elif conf >= 0.55:
        return REANALYZE_BUCKETS.get("medium", 2.0)
    return REANALYZE_BUCKETS.get("low", 1.0)

def bucket_label(snap):
    """Returns a string label representing the detection confidence bucket."""
    conf = snap.get("avg_conf", 0)
    if conf >= 0.80: return "high"
    if conf >= 0.55: return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
face_app = FaceAnalysis(allowed_modules=["detection", "genderage"])
face_app.prepare(
    ctx_id=0 if torch.cuda.is_available() else -1,
    det_size=INSIGHTFACE_DET
)
print("InsightFace ready")

workers_setup(face_app)
start_workers()

detector = Detector()

# ---------------------------------------------------------------------------
# Video + ads
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
effective_delay = FRAME_SKIP / fps

ads = {}
for key, path in AD_PATHS.items():
    img = cv2.imread(path)
    if img is not None:
        ads[key] = cv2.resize(img, DISPLAY_SIZE)
    else:
        print(f"[WARN] Could not load ad: {path}")

# ---------------------------------------------------------------------------
# Control state
# ---------------------------------------------------------------------------
frame_count      = 0
last_switch_time = 0.0
current_mode     = None

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
while True:
    frame_start = time.time()

    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    frame       = cv2.resize(frame, DISPLAY_SIZE)
    debug_frame = frame.copy()
    now         = time.time()

    # -----------------------------------------------------------------------
    # Detection + identity update
    # -----------------------------------------------------------------------
    detections   = detector.detect(frame)
    active_ids   = set()
    audience_ids = set()

    for det in detections:
        person_id = det["person_id"]
        conf      = det["conf"]
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        bbox_area = det["bbox_area"]

        active_ids.add(person_id)

        # 1. Update memory safely
        with memory_lock:
            if person_id not in identity_memory:
                mem = new_entry(now)
                # Inject missing tracking fields required by main loop
                mem["frames_seen"] = 1
                mem["avg_conf"]    = conf
                mem["facing_away"] = False
                mem["source"]      = "YOLO"
                identity_memory[person_id] = mem
            else:
                update_entry(identity_memory[person_id], conf, now)

        # 2. Grab a thread-safe snapshot for ALL logic and drawing
        snap = get_entry_snapshot(person_id)
        if not snap:
            continue

        tier               = get_analysis_tier(snap, bbox_area)
        last_analyzed      = snap["last_analyzed"]
        reanalyze_interval = get_reanalyze_interval(snap)
        facing_away        = snap.get("facing_away", False)

        # audience: stable + facing screen
        if is_stable(snap) and not facing_away:
            audience_ids.add(person_id)

        # queue for analysis
        should_analyze = tier != "noisy" and (now - last_analyzed) > reanalyze_interval

        # facing-away still retried at low bucket interval
        if facing_away and tier != "noisy":
            should_analyze = (now - last_analyzed) > REANALYZE_BUCKETS.get("low", 1.0)

        if should_analyze:
            # 1. Estimate head dimensions based on YOLO body box
            body_w = x2 - x1
            body_h = y2 - y1
            
            # Use ~35% of body height, capped by 80% of body width for close-ups
            est_face_size = int(min(body_h * 0.35, body_w * 0.8))
            est_face_size = max(est_face_size, 20)  # Safety minimum
            
            # 2. Target center of the head (X is center, Y is near the top)
            center_x = x1 + (body_w // 2)
            center_y = y1 + (est_face_size // 2)
            
            # 3. Create a generous square to guarantee capturing chin, hair, and context
            final_square_size = int(est_face_size * 1.5)
            half_size = final_square_size // 2
            
            # Ideal square coordinates (may theoretically fall outside camera bounds)
            cx1 = center_x - half_size
            cy1 = center_y - half_size
            cx2 = center_x + half_size
            cy2 = center_y + half_size
            
            # 4. Safe array slicing within frame boundaries
            frame_h, frame_w = frame.shape[:2]
            slice_y1 = max(0, cy1)
            slice_y2 = min(frame_h, cy2)
            slice_x1 = max(0, cx1)
            slice_x2 = min(frame_w, cx2)
            
            raw_crop = frame[slice_y1:slice_y2, slice_x1:slice_x2]
            
            if raw_crop.size > 0:
                # 5. Restore perfect 1:1 aspect ratio via black padding if cropped by edge
                pad_top    = max(0, -cy1)
                pad_bottom = max(0, cy2 - frame_h)
                pad_left   = max(0, -cx1)
                pad_right  = max(0, cx2 - frame_w)
                
                if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                    crop = cv2.copyMakeBorder(
                        raw_crop, pad_top, pad_bottom, pad_left, pad_right, 
                        cv2.BORDER_CONSTANT, value=(0, 0, 0)
                    )
                else:
                    crop = raw_crop.copy()
                
                # 6. Final geometric safety check and dispatch
                if crop.size > 0 and crop.shape[0] == crop.shape[1]:
                    if tier == "precision" and not facing_away:
                        enqueue_deepface(person_id, crop)
                    else:
                        enqueue_insightface(person_id, crop)
                    
                    with memory_lock:
                        if person_id in identity_memory:
                            identity_memory[person_id]["last_analyzed"] = now

        # -----------------------------------------------------------------------
        # Draw detection on debug frame (Using the exact same safe `snap`)
        # -----------------------------------------------------------------------
        age        = snap.get("age")
        gender     = snap.get("gender")
        source     = snap.get("source")
        emotion    = snap.get("emotion")
        blabel     = bucket_label(snap)

        # Dynamically determine demographic data source for display stability
        if emotion:
            source = "DF"
        elif age and gender and gender != "?":
            source = "IF"

        if facing_away:
            box_color = (50, 50, 150)
            label     = f"ID{person_id} | away"
            txt_color = (50, 50, 150)
            thickness = 1

        elif tier == "noisy":
            box_color = (80, 80, 80)
            label     = f"ID{person_id} | noisy"
            txt_color = (80, 80, 80)
            thickness = 1

        elif tier == "precision":
            box_color = (0, 165, 255)
            thickness = 2
            emo_tag   = f" {emotion}" if emotion else ""
            if age and gender and gender != "?":
                label = f"ID{person_id} | {gender} ~{age}{emo_tag} [{source}][{blabel}]"
            else:
                label = f"ID{person_id} | precision..."
            txt_color = (0, 165, 255)

        else: # Standard
            thickness = 2
            if age and gender:
                box_color = (0, 255, 0)
                label     = f"ID{person_id} | {gender} ~{age} [{source}][{blabel}]"
                txt_color = (0, 255, 0)
            else:
                box_color = (0, 200, 255)
                label     = f"ID{person_id} | stable..."
                txt_color = (0, 200, 255)

        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), box_color, thickness)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(debug_frame, (x1, y1 - th - 12), (x1 + tw + 6, y1), (0, 0, 0), -1)
        cv2.putText(debug_frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 1)

    # -----------------------------------------------------------------------
    # Cleanup stale IDs
    # -----------------------------------------------------------------------
    prune_stale_identities()

    # -----------------------------------------------------------------------
    # Mode decision (placeholder — decision engine replaces this next)
    # -----------------------------------------------------------------------
    people_count   = len(active_ids)
    audience_count = len(audience_ids)

    if audience_count == 0:
        new_mode = "empty"
    elif audience_count > 2:
        new_mode = "ad1"
    else:
        new_mode = "ad2"

    if new_mode != current_mode and (now - last_switch_time) > AD_COOLDOWN:
        current_mode     = new_mode
        last_switch_time = now

    # -----------------------------------------------------------------------
    # Ad display (placeholder — display module replaces this next)
    # -----------------------------------------------------------------------
    if current_mode == "empty":
        display = np.zeros((*DISPLAY_SIZE[::-1], 3), dtype=np.uint8)
        cv2.putText(display, "RESOURCE PRESERVING MODE",
                    (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    elif current_mode in ads:
        display = ads[current_mode].copy()
        cv2.putText(display, current_mode.upper(), (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    else:
        display = np.zeros((*DISPLAY_SIZE[::-1], 3), dtype=np.uint8)

    # -----------------------------------------------------------------------
    # Summary overlay on ad window
    # -----------------------------------------------------------------------
    all_mems        = get_snapshot().values()
    away_count      = sum(1 for m in all_mems if m.get("facing_away"))
    noisy_count     = sum(1 for m in all_mems if not is_stable(m) and not m.get("facing_away"))
    
    # Exclude targets that have escalated into the precision category to prevent dual-counting
    standard_count  = sum(1 for m in all_mems if is_stable(m) and not m.get("facing_away") and not (
                          m.get("frames_seen", 0) >= PRECISION_FRAMES_REQUIRED and 
                          m.get("avg_conf", 0) >= PRECISION_CONF_THRESHOLD))
    
    precision_count = sum(1 for m in all_mems
                          if m.get("frames_seen", 0) >= PRECISION_FRAMES_REQUIRED
                          and m.get("avg_conf", 0)   >= PRECISION_CONF_THRESHOLD
                          and not m.get("facing_away"))

    cv2.putText(display, f"Detected:  {people_count}",
                (10, 598), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(display, f"Audience:  {audience_count}",
                (10, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(display, f"Away:      {away_count}",
                (10, 642), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 150), 1)
    cv2.putText(display, f"Noisy:     {noisy_count}",
                (10, 664), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
    cv2.putText(display, f"Stable IF: {standard_count}",
                (10, 686), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
    cv2.putText(display, f"Prec  DF:  {precision_count}",
                (10, 708), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)

    # -----------------------------------------------------------------------
    # FPS sync
    # -----------------------------------------------------------------------
    elapsed    = time.time() - frame_start
    sleep_time = effective_delay - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)

    cv2.imshow("TRACKING (DEBUG)", debug_frame)
    cv2.imshow("AD DISPLAY", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()