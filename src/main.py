# =============================================================================
# main.py
# =============================================================================

import cv2
import numpy as np
import time
import torch
from insightface.app import FaceAnalysis

from config import (
    VIDEO_PATH, DISPLAY_SIZE, FRAME_SKIP, INSIGHTFACE_DET,
    REANALYZE_BUCKETS, AD_COOLDOWN, AD_PATHS,
    STABLE_FRAMES_REQUIRED, STABLE_CONF_THRESHOLD,
    PRECISION_FRAMES_REQUIRED, PRECISION_CONF_THRESHOLD, PRECISION_BBOX_AREA,
    FACING_AWAY_RECHECK_INTERVAL,
)
from vision.detector import Detector
from vision.identity import (
    identity_memory, memory_lock,
    new_entry, update_entry,
    get_analysis_tier, get_reanalyze_interval, is_stable,
    bucket_label, get_entry_snapshot, get_snapshot,
    prune_stale_identities,
)
from vision.workers import (
    setup as workers_setup, start_workers,
    enqueue_insightface, enqueue_deepface,
)

# ---------------------------------------------------------------------------
# Setup
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
if not cap.isOpened():
    print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
    exit(1)

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
last_prune_time  = 0.0
current_mode     = None

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
while True:
    frame_start = time.time()

    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    frame       = cv2.resize(frame, DISPLAY_SIZE)
    debug_frame = frame.copy()
    now         = time.time()

    detections   = detector.detect(frame)
    active_ids   = set()
    audience_ids = set()

    for det in detections:
        person_id       = det["person_id"]
        conf            = det["conf"]
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        bbox_area       = det["bbox_area"]

        active_ids.add(person_id)

        # -------------------------------------------------------------------
        # 1. Update identity memory
        # -------------------------------------------------------------------
        with memory_lock:
            if person_id not in identity_memory:
                identity_memory[person_id] = new_entry(conf, now, bbox_area)
            else:
                update_entry(identity_memory[person_id], conf, now, bbox_area)

            mem     = identity_memory[person_id]

            # movement tracking
            history = mem["area_history"]
            history.append(bbox_area)
            if len(history) > 30:
                history.pop(0)
            if len(history) >= 20:
                old_avg = sum(history[:10]) / 10
                new_avg = sum(history[-10:]) / 10
                if old_avg > new_avg * 1.15:
                    mem["movement"] = "away"
                elif new_avg > old_avg * 1.15:
                    mem["movement"] = "closer"
                else:
                    mem["movement"] = "stable"

            tier               = get_analysis_tier(mem, bbox_area)
            last_analyzed      = mem["last_analyzed"]
            reanalyze_interval = get_reanalyze_interval(mem)
            facing_away        = mem["facing_away"]

        # -------------------------------------------------------------------
        # 2. Audience — stable + not facing away
        # -------------------------------------------------------------------
        if is_stable(mem) and not facing_away:
            audience_ids.add(person_id)

        # -------------------------------------------------------------------
        # 3. Analysis gating — periodic recheck for facing-away
        # -------------------------------------------------------------------
        should_analyze = (
            tier != "noisy" and
            (now - last_analyzed) > reanalyze_interval and
            (not facing_away or
             (now - last_analyzed) > FACING_AWAY_RECHECK_INTERVAL)
        )

        if should_analyze:
            # Reliable 40% head crop — no pose dependency
            pad     = 10
            head_y2 = y1 + int((y2 - y1) * 0.40)
            cx1     = max(0, x1 - pad)
            cy1     = max(0, y1 - pad)
            cx2     = min(frame.shape[1], x2 + pad)
            cy2     = min(frame.shape[0], head_y2 + pad)
            crop    = frame[cy1:cy2, cx1:cx2].copy()

            if crop.size > 0:
                if tier == "precision":
                    enqueue_deepface(person_id, crop)
                else:
                    enqueue_insightface(person_id, crop)
                with memory_lock:
                    if person_id in identity_memory:
                        identity_memory[person_id]["last_analyzed"] = now

        # -------------------------------------------------------------------
        # 4. Draw
        # -------------------------------------------------------------------
        snap     = get_entry_snapshot(person_id)
        age      = snap.get("age")
        gender   = snap.get("gender")
        emotion  = snap.get("emotion")
        movement = snap.get("movement", "stable")
        blabel   = bucket_label(snap)
        source   = snap.get("source", "")
        facing_away_snap = snap.get("facing_away", False)
        move_tag = " [↓]" if movement == "away" else " [↑]" if movement == "closer" else ""

        if facing_away_snap:
            box_color = (50, 50, 150)
            txt_color = (50, 50, 150)
            label     = f"ID{person_id} | away{move_tag}"
            thickness = 1

        elif tier == "noisy":
            box_color = (80, 80, 80)
            txt_color = (80, 80, 80)
            label     = f"ID{person_id} | noisy{move_tag}"
            thickness = 1

        elif tier == "precision":
            box_color = (0, 165, 255)
            txt_color = (0, 165, 255)
            thickness = 2
            emo_tag   = f" {emotion}" if emotion else ""
            if age and gender and gender != "?":
                label = f"ID{person_id} | {gender} ~{age}{emo_tag} [{source}][{blabel}]{move_tag}"
            else:
                label = f"ID{person_id} | precision...{move_tag}"

        elif age and gender and gender != "?":
            box_color = (0, 255, 0)
            txt_color = (0, 255, 0)
            thickness = 2
            label     = f"ID{person_id} | {gender} ~{age} [{source}][{blabel}]{move_tag}"

        elif is_stable(snap):
            box_color = (0, 200, 255)
            txt_color = (0, 200, 255)
            thickness = 2
            label     = f"ID{person_id} | stable...{move_tag}"

        else:
            box_color = (80, 80, 80)
            txt_color = (80, 80, 80)
            thickness = 1
            label     = f"ID{person_id} | noisy{move_tag}"

        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), box_color, thickness)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(debug_frame, (x1, y1 - th - 12), (x1 + tw + 6, y1), (0, 0, 0), -1)
        cv2.putText(debug_frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 1)

    # -----------------------------------------------------------------------
    # 5. Cleanup (throttled — once per second)
    # -----------------------------------------------------------------------
    if now - last_prune_time > 1.0:
        prune_stale_identities()
        last_prune_time = now

    # -----------------------------------------------------------------------
    # 6. Mode decision (placeholder)
    # -----------------------------------------------------------------------
    audience_count = len(audience_ids)
    people_count   = len(active_ids)

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
    # 7. Ad display (placeholder)
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
    # 8. Dashboard
    # -----------------------------------------------------------------------
    snap_dict       = get_snapshot()
    noisy_count     = sum(1 for m in snap_dict.values() if not is_stable(m))
    away_count      = sum(1 for m in snap_dict.values()
                          if is_stable(m) and m.get("facing_away"))
    standard_count  = sum(1 for m in snap_dict.values()
                          if is_stable(m) and not m.get("facing_away")
                          and not (
                              m.get("frames_seen", 0) >= PRECISION_FRAMES_REQUIRED
                              and m.get("avg_conf", 0) >= PRECISION_CONF_THRESHOLD
                              and m.get("bbox_area", 0) >= PRECISION_BBOX_AREA
                          ))
    precision_count = sum(1 for m in snap_dict.values()
                          if is_stable(m) and not m.get("facing_away")
                          and m.get("frames_seen", 0) >= PRECISION_FRAMES_REQUIRED
                          and m.get("avg_conf", 0)    >= PRECISION_CONF_THRESHOLD
                          and m.get("bbox_area", 0)   >= PRECISION_BBOX_AREA)

    cv2.rectangle(debug_frame, (5, 570), (240, 715), (0, 0, 0), -1)
    cv2.putText(debug_frame, f"Detected:  {people_count}",
                (10, 590), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(debug_frame, f"Audience:  {audience_count}",
                (10, 610), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(debug_frame, f"Away:      {away_count}",
                (10, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 150), 1)
    cv2.putText(debug_frame, f"Noisy:     {noisy_count}",
                (10, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
    cv2.putText(debug_frame, f"Stable IF: {standard_count}",
                (10, 670), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    cv2.putText(debug_frame, f"Prec  DF:  {precision_count}",
                (10, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    # -----------------------------------------------------------------------
    # 9. FPS sync
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
