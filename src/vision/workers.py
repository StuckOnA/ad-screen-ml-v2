# =============================================================================
# vision/workers.py
#
# Background AI workers for InsightFace (Standard) and DeepFace (Precision).
# Implements robust gender mapping, resolution gating, and bounded age tracking.
# =============================================================================

import cv2
import queue
import threading
from deepface import DeepFace
from vision.identity import (
    register_insightface_result,
    register_deepface_result,
    register_no_face
)

# ---------------------------------------------------------------------------
# State & Queues
# ---------------------------------------------------------------------------
if_queue = queue.Queue(maxsize=50)
df_queue = queue.Queue(maxsize=50)
face_app_instance = None

def setup(app_instance):
    """Stores the InsightFace instance created in main.py"""
    global face_app_instance
    face_app_instance = app_instance

def enqueue_insightface(person_id: int, crop) -> None:
    if not if_queue.full():
        if_queue.put((person_id, crop))

def enqueue_deepface(person_id: int, crop) -> None:
    if not df_queue.full():
        df_queue.put((person_id, crop))

# ---------------------------------------------------------------------------
# Standard Tier: InsightFace Worker
# ---------------------------------------------------------------------------
def insightface_worker():
    """Fast, lightweight analysis for stable tracking targets."""
    while True:
        person_id, crop = if_queue.get()
        try:
            # 1. Resolution Gate: Reject crops too small for accurate age/gender math
            if crop.shape[0] < 30 or crop.shape[1] < 30:
                register_no_face(person_id)
                continue
                
            faces = face_app_instance.get(crop)
            if not faces:
                register_no_face(person_id)
                continue
                
            face = faces[0]  # Take the most prominent face in the tight crop
            
            # 2. Robust Gender Mapping
            # The standard mapping is 0 = Female, 1 = Male. 
            # If your specific model weights are still flipping the genders, 
            # simply swap the "F" and "M" strings below.
            INSIGHTFACE_GENDER_MAP = {
                0: "F",  
                1: "M"   
            }
            
            # Safely extract the attribute (handles different library versions)
            raw_gender = getattr(face, "gender", getattr(face, "sex", None))
            if raw_gender is not None:
                gender = INSIGHTFACE_GENDER_MAP.get(int(raw_gender), "?")
            else:
                gender = "?"
            
            # 3. Bounded Age Extraction
            raw_age = getattr(face, "age", None)
            if raw_age is not None and raw_age > 0:
                # Cap ages to realistic human bounds to prevent wild median skewing
                age = max(1, min(100, int(raw_age)))
            else:
                age = None
                
            register_insightface_result(person_id, gender, age, face.det_score)

        except Exception as e:
            register_no_face(person_id)
            print(f"[WARN] InsightFace worker error on ID {person_id}: {e}")
        finally:
            if_queue.task_done()

# ---------------------------------------------------------------------------
# Precision Tier: DeepFace Worker
# ---------------------------------------------------------------------------
def deepface_worker():
    """Heavyweight, high-accuracy analysis for primary audience members."""
    while True:
        person_id, crop = df_queue.get()
        try:
            # 1. Resolution Gate: DeepFace requires slightly more data for high accuracy
            if crop.shape[0] < 45 or crop.shape[1] < 45:
                register_no_face(person_id)
                continue
                
            # enforce_detection=False prevents DeepFace from crashing if its 
            # internal detector disagrees with YOLO's bounding box.
            results = DeepFace.analyze(
                img_path=crop,
                actions=['age', 'gender', 'emotion'],
                enforce_detection=False, 
                silent=True
            )
            
            # Unpack list if DeepFace returns multiple faces
            res = results[0] if isinstance(results, list) else results
            
            # 2. Version-Agnostic Gender Extraction
            gender_data = res.get("gender", {})
            if isinstance(gender_data, dict):
                # Cascading fallback through every key DeepFace has ever used
                w_score = gender_data.get("Woman", gender_data.get("Female", gender_data.get("female", 0)))
                m_score = gender_data.get("Man", gender_data.get("Male", gender_data.get("male", 0)))
                gender = "F" if w_score > m_score else "M"
            else:
                # Absolute fallback if a specific version returns just a raw string
                g_str = str(gender_data).lower()
                gender = "F" if "wom" in g_str or "fem" in g_str else "M"
                
            # 3. Bounded Age Extraction
            raw_age = res.get("age")
            age = max(1, min(100, int(raw_age))) if raw_age is not None else None
            
            # 4. Emotion Extraction
            emotion = res.get("dominant_emotion")
            emo_scores = res.get("emotion", {})
            conf = res.get("face_confidence", 1.0)
            
            register_deepface_result(person_id, gender, age, emotion, emo_scores, conf)

        except Exception as e:
            register_no_face(person_id)
            # Suppressing the exact error print here is recommended as DeepFace 
            # throws verbose ValueErrors frequently on blurry crops.
        finally:
            df_queue.task_done()

# ---------------------------------------------------------------------------
# Thread Initialization
# ---------------------------------------------------------------------------
def start_workers():
    """Spin up the daemon threads."""
    threading.Thread(target=insightface_worker, daemon=True, name="IF_Worker").start()
    threading.Thread(target=deepface_worker, daemon=True, name="DF_Worker").start()