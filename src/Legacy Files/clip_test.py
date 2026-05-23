import cv2
import torch
import clip
from PIL import Image
import numpy as np
from ultralytics import YOLO

# -----------------------
# Models
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# YOLO for person detection
yolo = YOLO("yolov8n.pt")
yolo.to(device)

# CLIP
print("Loading CLIP...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()
print("CLIP ready")

# -----------------------
# Labels — edit these freely to test
# -----------------------
CLOTHING_LABELS = [
    "borka",
    "panjabi",
    "blazer",
    "casual clothes",
    "sportswear",
    "school uniform",
    "suit",
    "saree",
    "jeans and t-shirt",
]

ACCESSORY_LABELS = [
    "backpack",
    "handbag",
    "umbrella",
    "helmet",
    "cap or hat",
    "no visible accessory",
]

# prompt template — wraps single labels for better CLIP accuracy
CLOTHING_TEMPLATE  = "a person wearing {}"
ACCESSORY_TEMPLATE = "a person carrying {}"

# tokenize once at startup
clothing_tokens  = clip.tokenize(
    [CLOTHING_TEMPLATE.format(l)  for l in CLOTHING_LABELS]
).to(device)

accessory_tokens = clip.tokenize(
    [ACCESSORY_TEMPLATE.format(l) for l in ACCESSORY_LABELS]
).to(device)

# -----------------------
# CLIP inference
# -----------------------
def classify_crop(crop_bgr):
    if crop_bgr.size == 0:
        return None, None, None, None

    img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = clip_preprocess(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        # clothing
        logits_c, _ = clip_model(img_tensor, clothing_tokens)
        probs_c     = logits_c.softmax(dim=-1).cpu().numpy()[0]

        # accessory
        logits_a, _ = clip_model(img_tensor, accessory_tokens)
        probs_a     = logits_a.softmax(dim=-1).cpu().numpy()[0]

    top_clothing   = CLOTHING_LABELS[probs_c.argmax()]
    top_clothing_conf = float(probs_c.max())

    top_accessory  = ACCESSORY_LABELS[probs_a.argmax()]
    top_accessory_conf = float(probs_a.max())

    return top_clothing, top_clothing_conf, top_accessory, top_accessory_conf

# -----------------------
# Video
# -----------------------
cap = cv2.VideoCapture("videos/test4.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
SIZE = (1280, 720)

frame_skip  = 2
frame_count = 0

# cache results per person so CLIP doesn't run every frame
result_cache = {}   # { person_id: { clothing, clothing_conf, accessory, accessory_conf } }
CACHE_REFRESH_SECONDS = 3.0
import time
last_analyzed = {}  # { person_id: timestamp }

# -----------------------
# Main loop
# -----------------------
while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    frame      = cv2.resize(frame, SIZE)
    debug      = frame.copy()
    now        = time.time()

    results = yolo.track(frame, classes=[0], persist=True, device=device)
    boxes   = results[0].boxes

    if boxes is not None and boxes.id is not None:
        ids    = boxes.id.int().tolist()
        confs  = boxes.conf.tolist()
        coords = boxes.xyxy.tolist()

        for person_id, conf, box in zip(ids, confs, coords):
            if conf < 0.4:
                continue

            x1, y1, x2, y2 = map(int, box)

            # full body crop for CLIP (not head crop — needs whole outfit)
            pad = 10
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(frame.shape[1], x2 + pad)
            cy2 = min(frame.shape[0], y2 + pad)
            crop = frame[cy1:cy2, cx1:cx2]

            # run CLIP if cache is stale
            due = (person_id not in last_analyzed or
                   now - last_analyzed[person_id] > CACHE_REFRESH_SECONDS)

            if due and crop.size > 0:
                clothing, clothing_conf, accessory, accessory_conf = classify_crop(crop)
                result_cache[person_id] = {
                    "clothing":       clothing,
                    "clothing_conf":  clothing_conf,
                    "accessory":      accessory,
                    "accessory_conf": accessory_conf,
                }
                last_analyzed[person_id] = now

            cached = result_cache.get(person_id, {})
            clothing      = cached.get("clothing",       "analyzing...")
            clothing_conf = cached.get("clothing_conf",  0.0)
            accessory     = cached.get("accessory",      "")
            accessory_conf = cached.get("accessory_conf", 0.0)

            # --- Draw ---
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # clothing label
            c_label = f"{clothing} ({clothing_conf:.0%})" if clothing_conf > 0 else "analyzing..."
            (tw, th), _ = cv2.getTextSize(c_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(debug, (x1, y1 - th - 12), (x1 + tw + 6, y1), (0, 0, 0), -1)
            cv2.putText(debug, c_label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # accessory label below bbox
            if accessory and accessory != "no visible accessory":
                a_label = f"+ {accessory} ({accessory_conf:.0%})"
                (aw, ah), _ = cv2.getTextSize(a_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(debug, (x1, y2), (x1 + aw + 6, y2 + ah + 10), (0, 0, 0), -1)
                cv2.putText(debug, a_label, (x1 + 3, y2 + ah + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

            # confidence bar for clothing
            bar_w = x2 - x1
            filled = int(bar_w * clothing_conf)
            bar_color = (0, 255, 0) if clothing_conf > 0.5 else (0, 165, 255) if clothing_conf > 0.3 else (0, 0, 255)
            cv2.rectangle(debug, (x1, y2 + 25), (x1 + bar_w, y2 + 32), (40, 40, 40), -1)
            cv2.rectangle(debug, (x1, y2 + 25), (x1 + filled, y2 + 32), bar_color, -1)

    # --- Legend ---
    cv2.putText(debug, "CLIP CLOTHING TEST", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(debug, "Green bar = clothing confidence",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(debug, "Cyan label = accessory",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    cv2.imshow("CLIP CLOTHING TEST", debug)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()