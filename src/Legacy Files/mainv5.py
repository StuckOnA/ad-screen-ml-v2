import cv2
import numpy as np
import time
import torch
import threading
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from deepface import DeepFace

# -----------------------
# Models
# -----------------------
model = YOLO("yolov8n.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("Using device:", device)

face_app = FaceAnalysis(allowed_modules=["detection", "genderage"])
face_app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(320, 320))
print("InsightFace ready")

# -----------------------
# Video
# -----------------------
cap = cv2.VideoCapture("videos/test9.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
effective_delay = 2 / fps

# -----------------------
# Ads
# -----------------------
ad1 = cv2.imread("ads/ad1.jpg")
ad2 = cv2.imread("ads/ad2.jpg")
SIZE = (1280, 720)
ad1 = cv2.resize(ad1, SIZE)
ad2 = cv2.resize(ad2, SIZE)

# -----------------------
# Identity Memory
# -----------------------
identity_memory = {}
memory_lock = threading.Lock()

pending_insightface = {}
pending_deepface    = {}
pending_if_lock     = threading.Lock()
pending_df_lock     = threading.Lock()

# -----------------------
# Tier config
# -----------------------
STABLE_FRAMES_REQUIRED     = 15
STABLE_CONF_THRESHOLD      = 0.55

PRECISION_FRAMES_REQUIRED  = 40
PRECISION_CONF_THRESHOLD   = 0.75
PRECISION_BBOX_AREA        = 45000

FACING_AWAY_MISS_THRESHOLD         = 3
FACING_AWAY_INITIAL_THRESHOLD      = 1

# -----------------------
# Bucket config
# -----------------------
REANALYZE_BUCKETS = {
    "high":   15.0,
    "medium":  7.0,
    "low":     2.0,
}

def get_reanalyze_interval(mem):
    score = mem.get("result_confidence", None)
    if score is None:
        return 1.0
    elif score >= 0.80:
        return REANALYZE_BUCKETS["high"]
    elif score >= 0.55:
        return REANALYZE_BUCKETS["medium"]
    else:
        return REANALYZE_BUCKETS["low"]

def is_stable(mem):
    return (
        mem["frames_seen"] >= STABLE_FRAMES_REQUIRED and
        mem["avg_conf"]    >= STABLE_CONF_THRESHOLD
    )

def get_analysis_tier(mem, bbox_area):
    if (
        mem["frames_seen"] >= PRECISION_FRAMES_REQUIRED and
        mem["avg_conf"]    >= PRECISION_CONF_THRESHOLD  and
        bbox_area          >= PRECISION_BBOX_AREA
    ):
        return "precision"
    elif is_stable(mem):
        return "standard"
    else:
        return "noisy"

# -----------------------
# InsightFace worker
# -----------------------
def insightface_worker():
    while True:
        time.sleep(0.05)

        with pending_if_lock:
            if not pending_insightface:
                continue
            person_id, payload = next(iter(pending_insightface.items()))
            del pending_insightface[person_id]

        crop = payload["crop"]

        try:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            faces    = face_app.get(crop_rgb)

            if not faces:
                with memory_lock:
                    if person_id in identity_memory:
                        mem           = identity_memory[person_id]
                        never_labeled = mem["gender"] is None
                        threshold     = FACING_AWAY_INITIAL_THRESHOLD if never_labeled else FACING_AWAY_MISS_THRESHOLD
                        mem["result_confidence"]   = 0.0
                        mem["consecutive_misses"] += 1
                        if mem["consecutive_misses"] >= threshold:
                            mem["facing_away"] = True
                continue

            face              = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            gender            = "M" if face.gender == 1 else "F"
            age               = int(face.age)
            result_confidence = float(face.det_score)

            if not (5 <= age <= 90):
                age = None

            with memory_lock:
                if person_id in identity_memory:
                    mem                        = identity_memory[person_id]
                    mem["gender"]              = gender
                    mem["age"]                 = age
                    mem["source"]              = "IF"
                    mem["result_confidence"]   = result_confidence
                    mem["consecutive_misses"]  = 0
                    mem["facing_away"]         = False

            bucket = "high" if result_confidence >= 0.80 else "medium" if result_confidence >= 0.55 else "low"
            print(f"[IF] ID{person_id} | {gender} ~{age} | conf={result_confidence:.2f} → {bucket}")

        except Exception as e:
            print(f"[IF worker] ID{person_id}: {e}")

# -----------------------
# DeepFace worker
# -----------------------
def deepface_worker():
    while True:
        time.sleep(0.05)

        with pending_df_lock:
            if not pending_deepface:
                continue
            person_id, payload = next(iter(pending_deepface.items()))
            del pending_deepface[person_id]

        crop = payload["crop"]

        try:
            result = DeepFace.analyze(
                crop,
                actions=["age", "gender", "emotion"],
                enforce_detection=False,
                detector_backend="retinaface",
                silent=True
            )
            r             = result[0] if isinstance(result, list) else result
            age           = r.get("age", None)
            gender_scores = r.get("gender", {})
            woman_score   = gender_scores.get("Woman", 0)
            man_score     = gender_scores.get("Man", 0)
            gap           = abs(woman_score - man_score) / 100.0

            if gap < 0.15:
                gender = "?"
            elif woman_score > man_score:
                gender = "F"
            else:
                gender = "M"

            emotion           = r.get("dominant_emotion", None)
            emotion_scores    = r.get("emotion", {})
            result_confidence = gap

            with memory_lock:
                if person_id in identity_memory:
                    mem                       = identity_memory[person_id]
                    mem["gender"]             = gender
                    mem["age"]                = int(age) if age and (5 <= int(age) <= 90) else None
                    mem["source"]             = "DF"
                    mem["result_confidence"]  = result_confidence
                    mem["emotion"]            = emotion
                    mem["emotion_scores"]     = emotion_scores
                    mem["consecutive_misses"] = 0
                    mem["facing_away"]        = False

            bucket = "high" if result_confidence >= 0.80 else "medium" if result_confidence >= 0.55 else "low"
            print(f"[DF] ID{person_id} | {gender} ~{age} | {emotion} | conf={result_confidence:.2f} → {bucket}")

        except Exception as e:
            print(f"[DF worker] ID{person_id}: {e}")

threading.Thread(target=insightface_worker, daemon=True).start()
threading.Thread(target=deepface_worker,    daemon=True).start()

# -----------------------
# Control
# -----------------------
frame_skip       = 2
frame_count      = 0
last_switch_time = 0
cooldown         = 2
current_mode     = None

# -----------------------
# Main loop
# -----------------------
while True:
    frame_start = time.time()

    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    frame   = cv2.resize(frame, SIZE)
    results = model.track(frame, classes=[0], persist=True, device=device)

    active_ids   = set()
    audience_ids = set()
    debug_frame  = frame.copy()
    boxes        = results[0].boxes
    now          = time.time()

    if boxes is not None and boxes.id is not None:
        ids    = boxes.id.int().tolist()
        confs  = boxes.conf.tolist()
        coords = boxes.xyxy.tolist()

        for person_id, conf, box in zip(ids, confs, coords):
            if conf < 0.4:
                continue

            active_ids.add(person_id)
            x1, y1, x2, y2 = map(int, box)
            bbox_area       = (x2 - x1) * (y2 - y1)

            # --- Update identity memory ---
            with memory_lock:
                if person_id not in identity_memory:
                    identity_memory[person_id] = {
                        "age":               None,
                        "gender":            None,
                        "source":            None,
                        "emotion":           None,
                        "emotion_scores":    {},
                        "result_confidence": None,
                        "confidence":        conf,
                        "avg_conf":          conf,
                        "frames_seen":       1,
                        "last_seen":         now,
                        "last_analyzed":     0,
                        "consecutive_misses": 0,
                        "facing_away":       False,
                    }
                else:
                    mem               = identity_memory[person_id]
                    mem["last_seen"]   = now
                    mem["confidence"]  = conf
                    mem["frames_seen"] += 1
                    mem["avg_conf"]    = mem["avg_conf"] * 0.9 + conf * 0.1

                mem                = identity_memory[person_id]
                tier               = get_analysis_tier(mem, bbox_area)
                last_analyzed      = mem["last_analyzed"]
                reanalyze_interval = get_reanalyze_interval(mem)
                facing_away        = mem["facing_away"]

            # --- Track audience ---
            with memory_lock:
                cur_mem = identity_memory.get(person_id, {})
            if is_stable(cur_mem) and not cur_mem.get("facing_away", False):
                audience_ids.add(person_id)

            # --- Queue for analysis ---
            should_analyze = (
                tier != "noisy" and
                (now - last_analyzed) > reanalyze_interval
            )

            # facing-away targets still get retried at low interval
            if facing_away and tier != "noisy":
                should_analyze = (now - last_analyzed) > REANALYZE_BUCKETS["low"]

            if should_analyze:
                pad     = 20
                head_y2 = y1 + int((y2 - y1) * 0.40)
                cx1     = max(0, x1 - pad)
                cy1     = max(0, y1 - pad)
                cx2     = min(frame.shape[1], x2 + pad)
                cy2     = min(frame.shape[0], head_y2 + pad)
                crop    = frame[cy1:cy2, cx1:cx2].copy()

                if crop.size > 0:
                    if tier == "precision" and not facing_away:
                        with pending_df_lock:
                            pending_deepface[person_id] = {"crop": crop}
                    else:
                        with pending_if_lock:
                            pending_insightface[person_id] = {"crop": crop}
                    with memory_lock:
                        identity_memory[person_id]["last_analyzed"] = now

            # --- Read state for drawing ---
            with memory_lock:
                mem               = identity_memory.get(person_id, {})
                age               = mem.get("age")
                gender            = mem.get("gender")
                source            = mem.get("source")
                emotion           = mem.get("emotion")
                result_confidence = mem.get("result_confidence")
                facing_away       = mem.get("facing_away", False)
                tier              = get_analysis_tier(mem, bbox_area)

            bucket_label = (
                "?" if result_confidence is None else
                "H" if result_confidence >= 0.80 else
                "M" if result_confidence >= 0.55 else
                "L"
            )

            # --- Draw ---
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
                    label = f"ID{person_id} | {gender} ~{age}{emo_tag} [{source}][{bucket_label}]"
                else:
                    label = f"ID{person_id} | precision..."
                txt_color = (0, 165, 255)

            else:
                thickness = 2
                if age and gender:
                    box_color = (0, 255, 0)
                    label     = f"ID{person_id} | {gender} ~{age} [{source}][{bucket_label}]"
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

    # --- Clean stale IDs ---
    with memory_lock:
        stale = [pid for pid, m in identity_memory.items()
                 if now - m["last_seen"] > 5]
        for pid in stale:
            del identity_memory[pid]

    people_count   = len(active_ids)
    audience_count = len(audience_ids)

    # -----------------------
    # Mode decision
    # -----------------------
    if audience_count == 0:
        new_mode = "empty"
    elif audience_count > 2:
        new_mode = "ad1"
    else:
        new_mode = "ad2"

    if new_mode != current_mode and (now - last_switch_time) > cooldown:
        current_mode     = new_mode
        last_switch_time = now

    # -----------------------
    # Ad display
    # -----------------------
    if current_mode == "empty":
        display = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(display, "RESOURCE PRESERVING MODE",
                    (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    elif current_mode == "ad1":
        display = ad1.copy()
        cv2.putText(display, "AD 1 (crowd mode)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    elif current_mode == "ad2":
        display = ad2.copy()
        cv2.putText(display, "AD 2 (low crowd)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        display = np.zeros((360, 640, 3), dtype=np.uint8)

    # --- Summary ---
    with memory_lock:
        all_mems = list(identity_memory.values())

    away_count      = sum(1 for m in all_mems if m.get("facing_away"))
    noisy_count     = sum(1 for m in all_mems if not is_stable(m) and not m.get("facing_away"))
    standard_count  = sum(1 for m in all_mems if is_stable(m) and not m.get("facing_away"))
    precision_count = sum(1 for m in all_mems
                          if m["frames_seen"] >= PRECISION_FRAMES_REQUIRED
                          and m["avg_conf"]   >= PRECISION_CONF_THRESHOLD
                          and not m.get("facing_away"))
    high_conf       = sum(1 for m in all_mems if (m.get("result_confidence") or 0) >= 0.80)
    low_conf        = sum(1 for m in all_mems
                          if m.get("result_confidence") is not None
                          and m["result_confidence"] < 0.55)

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

    # -----------------------
    # FPS sync
    # -----------------------
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