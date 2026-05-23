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
context_model = YOLO("yolov8n.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
context_model.to(device)

print("Using device:", device)

face_app = FaceAnalysis(allowed_modules=["detection", "genderage"])
face_app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(320, 320))

# COCO context classes
CONTEXT_CLASSES = {
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    63: "laptop",
    67: "cellphone",
    39: "bottle",
    41: "cup",
    56: "chair",
    1: "bicycle",
    3: "motorcycle",
    32: "sportsball",
    36: "skateboard",
    48: "sandwich",
    52: "hotdog",
    53: "pizza",
}

# -----------------------
# IoU
# -----------------------
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = a1 + a2 - inter
    return inter / union if union > 0 else 0

# -----------------------
# Video
# -----------------------
cap = cv2.VideoCapture("videos/test9.mp4")
SIZE = (1280, 720)

# -----------------------
# Memory
# -----------------------
identity_memory = {}
memory_lock = threading.Lock()

pending_insightface = {}
pending_deepface = {}
pending_if_lock = threading.Lock()
pending_df_lock = threading.Lock()

# -----------------------
# Workers
# -----------------------
def insightface_worker():
    while True:
        time.sleep(0.05)

        with pending_if_lock:
            if not pending_insightface:
                continue
            person_id, crop = list(pending_insightface.items())[0]
            del pending_insightface[person_id]

        try:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            faces = face_app.get(crop_rgb)

            if not faces:
                continue

            face = faces[0]

            with memory_lock:
                if person_id in identity_memory:
                    identity_memory[person_id]["gender"] = "M" if face.gender == 1 else "F"
                    identity_memory[person_id]["age"] = int(face.age)
                    identity_memory[person_id]["source"] = "IF"

        except:
            pass


def deepface_worker():
    while True:
        time.sleep(0.05)

        with pending_df_lock:
            if not pending_deepface:
                continue
            person_id, crop = list(pending_deepface.items())[0]
            del pending_deepface[person_id]

        try:
            result = DeepFace.analyze(
                crop,
                actions=["emotion"],
                enforce_detection=False,
                silent=True
            )

            r = result[0] if isinstance(result, list) else result

            with memory_lock:
                if person_id in identity_memory:
                    identity_memory[person_id]["emotion"] = r.get("dominant_emotion")

        except:
            pass


threading.Thread(target=insightface_worker, daemon=True).start()
threading.Thread(target=deepface_worker, daemon=True).start()

# -----------------------
# Main loop
# -----------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, SIZE)

    results = model.track(
        frame,
        classes=[0],
        persist=True,
        device=device,
        verbose=False,
    )

    context_results = context_model(
        frame,
        classes=list(CONTEXT_CLASSES.keys()),
        device=device,
        verbose=False,
    )

    debug_frame = frame.copy()

    # =======================
    # CONTEXT OBJECTS
    # =======================
    detected_context = []

    ctx_boxes = context_results[0].boxes

    if ctx_boxes is not None:
        for box, cls, conf in zip(
            ctx_boxes.xyxy.tolist(),
            ctx_boxes.cls.tolist(),
            ctx_boxes.conf.tolist()
        ):
            cls = int(cls)

            if conf < 0.35 or cls not in CONTEXT_CLASSES:
                continue

            label = CONTEXT_CLASSES[cls]
            detected_context.append({"box": box, "label": label})

            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(
                debug_frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                1,
            )

    # =======================
    # PEOPLE
    # =======================
    boxes = results[0].boxes

    if boxes is not None and boxes.id is not None:

        ids = boxes.id.int().tolist()
        confs = boxes.conf.tolist()
        coords = boxes.xyxy.tolist()

        for person_id, conf, box in zip(ids, confs, coords):

            if conf < 0.4:
                continue

            x1, y1, x2, y2 = map(int, box)
            person_box = [x1, y1, x2, y2]

            attached = set()

            for obj in detected_context:
                if compute_iou(person_box, obj["box"]) > 0.25:
                    attached.add(obj["label"])

            with memory_lock:
                if person_id not in identity_memory:
                    identity_memory[person_id] = {
                        "age": None,
                        "gender": None,
                        "emotion": None,
                        "source": None,
                        "context": set(),
                    }

                identity_memory[person_id]["context"] = attached
                mem = identity_memory[person_id]

            h = y2 - y1
            crop = frame[max(0, y1):max(0, y1 + int(h * 0.4)), x1:x2].copy()

            if crop.size > 0:
                with pending_if_lock:
                    pending_insightface[person_id] = crop
                with pending_df_lock:
                    pending_deepface[person_id] = crop

            ctx = ",".join(sorted(attached))
            label = f"ID{person_id}"

            if mem.get("gender") and mem.get("age"):
                label += f" | {mem['gender']} ~{mem['age']}"

            if ctx:
                label += f" | {ctx}"

            if mem.get("emotion"):
                label += f" | {mem['emotion']}"

            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                debug_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    cv2.imshow("TRACKING (DEBUG)", debug_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()