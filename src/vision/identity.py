# =============================================================================
# vision/identity.py
#
# Thread-safe memory manager for tracked identities.
# Implements Temporal Ensembling (Median/Mode filtering) to stabilize 
# demographic and emotional predictions over time.
# =============================================================================

import time
import threading
import statistics
import copy
from collections import Counter

# ---------------------------------------------------------------------------
# Constants & Shared State
# ---------------------------------------------------------------------------
MAX_HISTORY = 15  # Number of recent valid frames to keep for smoothing

identity_memory: dict = {}
memory_lock = threading.Lock()  # Renamed back to match main.py


# ---------------------------------------------------------------------------
# Initialization & State Management
# ---------------------------------------------------------------------------

def new_entry(now: float) -> dict:
    """Initializes a blank state for a newly detected person."""
    return {
        # Spatial Tracking Data
        "bbox": None,
        "conf": 0.0,
        "last_seen": now,           # Updated continuously by YOLO
        "last_analyzed": 0.0,       # Updated by background AI workers
        
        # Smoothed Display Fields (Read by the main UI loop)
        "age": None,
        "gender": "?",
        "emotion": None,
        
        # Temporal Buffers (Used for statistical smoothing)
        "history_age": [],
        "history_gender": [],
        "history_emotion": [],
    }


def _apply_temporal_smoothing(mem: dict) -> None:
    """
    Applies statistical filters to history buffers to stabilize predictions.
    Instantly removes single-frame anomalies and flickering.
    """
    # 1. Gender: Majority Vote (Mode)
    if mem["history_gender"]:
        valid_genders = [g for g in mem["history_gender"] if g in ("M", "F")]
        if valid_genders:
            mem["gender"] = Counter(valid_genders).most_common(1)[0][0]

    # 2. Age: Median Filter
    if mem["history_age"]:
        valid_ages = [a for a in mem["history_age"] if a is not None]
        if valid_ages:
            mem["age"] = int(statistics.median(valid_ages))

    # 3. Emotion: Majority Vote (Mode)
    if "history_emotion" in mem and mem["history_emotion"]:
        valid_emotions = [e for e in mem["history_emotion"] if e is not None]
        if valid_emotions:
            mem["emotion"] = Counter(valid_emotions).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Background Worker Endpoints (Write Access)
# ---------------------------------------------------------------------------

def register_insightface_result(person_id: int, gender: str, age: int, det_score: float) -> None:
    """Called by the standard-tier background worker."""
    with memory_lock:
        if person_id not in identity_memory:
            return
            
        mem = identity_memory[person_id]
        
        # Append new valid data
        if gender in ("M", "F"):
            mem["history_gender"].append(gender)
        if age is not None:
            mem["history_age"].append(age)
            
        # Truncate to rolling window to prevent memory leaks
        mem["history_gender"] = mem["history_gender"][-MAX_HISTORY:]
        mem["history_age"]    = mem["history_age"][-MAX_HISTORY:]
        
        # Apply filters to update display fields
        _apply_temporal_smoothing(mem)
        
        mem["last_analyzed"] = time.time()


def register_deepface_result(person_id: int, gender: str, age: int, emotion: str, 
                             emotion_scores: dict, confidence: float) -> None:
    """Called by the precision-tier background worker."""
    with memory_lock:
        if person_id not in identity_memory:
            return
            
        mem = identity_memory[person_id]
        
        # Append new valid data
        if gender in ("M", "F"):
            mem["history_gender"].append(gender)
        if age is not None:
            mem["history_age"].append(age)
        if emotion is not None:
            mem["history_emotion"].append(emotion)
            
        # Truncate to rolling window
        mem["history_gender"]  = mem["history_gender"][-MAX_HISTORY:]
        mem["history_age"]     = mem["history_age"][-MAX_HISTORY:]
        mem["history_emotion"] = mem["history_emotion"][-MAX_HISTORY:]
        
        # Apply filters to update display fields
        _apply_temporal_smoothing(mem)
        
        mem["last_analyzed"] = time.time()


def register_no_face(person_id: int) -> None:
    """Registers that a worker failed to detect a valid face in the crop."""
    with memory_lock:
        if person_id in identity_memory:
            # We update the timestamp so we don't spam the worker with bad crops
            identity_memory[person_id]["last_analyzed"] = time.time()


# ---------------------------------------------------------------------------
# Main Loop Tracking & Lifecycle Endpoints
# ---------------------------------------------------------------------------

def update_bbox(person_id: int, bbox: list, conf: float, now: float) -> None:
    """Called continuously by the YOLO tracker in the main loop."""
    with memory_lock:
        if person_id not in identity_memory:
            identity_memory[person_id] = new_entry(now)
            
        mem = identity_memory[person_id]
        mem["bbox"] = bbox
        mem["conf"] = conf
        mem["last_seen"] = now


def get_snapshot() -> dict:
    """
    Returns a deep copy of the memory. 
    Crucial for allowing the main UI thread to draw bounding boxes and 
    labels without freezing up waiting on worker thread locks.
    """
    with memory_lock:
        return copy.deepcopy(identity_memory)


def prune_stale_identities(timeout_seconds: float = 2.0) -> None:
    """Removes IDs that have left the camera view to free up memory."""
    now = time.time()
    with memory_lock:
        stale_ids = [
            pid for pid, mem in identity_memory.items() 
            if (now - mem["last_seen"]) > timeout_seconds
        ]
        for pid in stale_ids:
            del identity_memory[pid]