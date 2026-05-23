# =============================================================================
# vision/detector.py
#
# Wraps YOLO tracking.
# Returns structured detection results per frame.
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

    def detect(self, frame) -> list[dict]:
        """
        Run YOLO tracking on a frame.
        Returns list of dicts:
            { person_id, conf, x1, y1, x2, y2, bbox_area }
        Only persons above MIN_DETECTION_CONF are returned.
        """
        # SAFE OPTIMIZATIONS ONLY: 
        # 1. Native conf filtering
        # 2. verbose=False to save I/O overhead
        results = self.model.track(
            frame,
            classes=[0],                  
            conf=MIN_DETECTION_CONF,      
            persist=True,
            device=self.device,
            verbose=False                 
        )

        detections = []
        boxes = results[0].boxes

        # Safety check for lost tracks / empty frames
        if boxes is None or boxes.id is None:
            return detections

        ids    = boxes.id.int().tolist()
        confs  = boxes.conf.tolist()
        coords = boxes.xyxy.tolist()

        for person_id, conf, box in zip(ids, confs, coords):
            x1, y1, x2, y2 = map(int, box)
            
            detections.append({
                "person_id": person_id,
                "conf":      conf,
                "x1":        x1,
                "y1":        y1,
                "x2":        x2,
                "y2":        y2,
                "bbox_area": (x2 - x1) * (y2 - y1),
            })

        return detections