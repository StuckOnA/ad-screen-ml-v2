# =============================================================================
# vision/detector.py
# =============================================================================

import torch
from ultralytics import YOLO
from config import YOLO_MODEL, MIN_DETECTION_CONF


class Detector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = YOLO(YOLO_MODEL)
        self.model.to(self.device)
        # MiVOLO's YOLO has 2 classes: 0=person, 1=face
        print(f"YOLO model loaded: {YOLO_MODEL}")
        print(f"  Classes: {self.model.names}")

    def detect(self, frame) -> tuple[list[dict], list[dict]]:
        """
        Run YOLO tracking on a frame.
        Returns:
            persons: list of {person_id, conf, x1, y1, x2, y2, bbox_area}
            faces:   list of {face_id, conf, x1, y1, x2, y2}
        """
        results = self.model.track(
            frame,
            conf=MIN_DETECTION_CONF,
            persist=True,
            device=self.device,
            verbose=False,
        )

        persons = []
        faces   = []
        boxes   = results[0].boxes

        if boxes is None or boxes.id is None:
            return persons, faces

        ids    = boxes.id.int().tolist()
        confs  = boxes.conf.tolist()
        coords = boxes.xyxy.tolist()
        classes = boxes.cls.int().tolist()

        for track_id, conf, box, cls in zip(ids, confs, coords, classes):
            x1, y1, x2, y2 = map(int, box)
            if cls == 0:  # person
                persons.append({
                    "person_id": track_id,
                    "conf":      conf,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "bbox_area": (x2 - x1) * (y2 - y1),
                })
            elif cls == 1:  # face
                faces.append({
                    "face_id": track_id,
                    "conf":    conf,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })

        return persons, faces


def associate_faces_to_persons(persons: list[dict], faces: list[dict]) -> dict:
    """
    Associate each person with their best-matching face detection.
    Uses: face center must be inside person bbox. Closest center wins.

    Returns:
        {person_id: face_dict_or_None}
    """
    associations = {p["person_id"]: None for p in persons}

    for face in faces:
        fx1, fy1, fx2, fy2 = face["x1"], face["y1"], face["x2"], face["y2"]
        face_cx = (fx1 + fx2) / 2
        face_cy = (fy1 + fy2) / 2

        best_pid  = None
        best_dist = float("inf")

        for person in persons:
            px1, py1, px2, py2 = person["x1"], person["y1"], person["x2"], person["y2"]
            if px1 <= face_cx <= px2 and py1 <= face_cy <= py2:
                pcx  = (px1 + px2) / 2
                pcy  = (py1 + py2) / 2
                dist = abs(face_cx - pcx) + abs(face_cy - pcy)
                if dist < best_dist:
                    best_dist = dist
                    best_pid  = person["person_id"]

        if best_pid is not None:
            existing = associations[best_pid]
            if existing is None or face["conf"] > existing["conf"]:
                associations[best_pid] = face

    return associations
