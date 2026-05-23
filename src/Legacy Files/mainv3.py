import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO

# -----------------------
# Model
# -----------------------
model = YOLO("yolov8n.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("Using device:", device)

# -----------------------
# Video
# -----------------------
cap = cv2.VideoCapture("videos/test.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30
delay = 1 / fps

# -----------------------
# Ads
# -----------------------
ad1 = cv2.imread("ads/ad1.jpg")
ad2 = cv2.imread("ads/ad2.jpg")

SIZE = (1280, 720)
ad1 = cv2.resize(ad1, SIZE)
ad2 = cv2.resize(ad2, SIZE)

# -----------------------
# Control
# -----------------------
frame_skip = 2
frame_count = 0

prev_time = time.time()

# cooldown logic
last_switch_time = 0
cooldown = 2  # seconds

current_mode = None  # "ad1" or "ad2" or "empty"

# -----------------------
# Main loop
# -----------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % frame_skip != 0:
        continue

    frame = cv2.resize(frame, SIZE)

    # -----------------------
    # tracking
    # -----------------------
    results = model.track(
        frame,
        classes=[0],
        persist=True,
        device=device
    )

    active_ids = set()

    boxes = results[0].boxes

    debug_frame = frame.copy()

    if boxes is not None and boxes.id is not None:
        ids = boxes.id.int().tolist()
        confs = boxes.conf.tolist()
        coords = boxes.xyxy.tolist()

        for i, conf, box in zip(ids, confs, coords):
            if conf >= 0.4:
                active_ids.add(i)

                # draw bbox (DEBUG WINDOW)
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(debug_frame, f"ID {i} {conf:.2f}",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1)

    people_count = len(active_ids)

    # -----------------------
    # MODE DECISION + COOLDOWN
    # -----------------------
    now = time.time()

    if people_count == 0:
        new_mode = "empty"
    elif people_count > 2:
        new_mode = "ad1"
    else:
        new_mode = "ad2"

    # apply cooldown
    if new_mode != current_mode and (now - last_switch_time) > cooldown:
        current_mode = new_mode
        last_switch_time = now

    # -----------------------
    # AD WINDOW
    # -----------------------
    if current_mode == "empty":
        display = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(display,
                    "RESOURCE PRESERVING MODE",
                    (40, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2)

    elif current_mode == "ad1":
        display = ad1.copy()
        cv2.putText(display, "AD 1 (crowd mode)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

    elif current_mode == "ad2":
        display = ad2.copy()
        cv2.putText(display, "AD 2 (low crowd)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

    else:
        display = np.zeros((360, 640, 3), dtype=np.uint8)

    # -----------------------
    # overlays
    # -----------------------
    cv2.putText(display,
                f"People: {people_count}",
                (10, 331),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    # -----------------------
    # FPS sync
    # -----------------------
    current_time = time.time()
    elapsed = current_time - prev_time

    if elapsed < delay:
        time.sleep(delay - elapsed)

    prev_time = time.time()

    # -----------------------
    # TWO WINDOWS
    # -----------------------
    cv2.imshow("TRACKING (DEBUG)", debug_frame)
    cv2.imshow("AD DISPLAY", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()