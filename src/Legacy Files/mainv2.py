import cv2
from ultralytics import YOLO
import time

# Load model
model = YOLO("yolov8n.pt")

# Use GPU if available
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Using device:", device)

cap = cv2.VideoCapture("videos/test.mp4")

frame_skip = 2   # process every 2nd frame
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # --- FRAME SKIP ---
    if frame_count % frame_skip != 0:
        continue

    # --- RESIZE (speed boost) ---
    frame = cv2.resize(frame, (1280, 720))

    start = time.time()

    # --- INFERENCE ---
    results = model.track(
        frame,
        classes=[0],   # person only
        persist=True,
        device=device
    )

    annotated = results[0].plot()

    # FPS display
    fps = 1 / (time.time() - start)
    cv2.putText(
        annotated,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("ad-screen", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()