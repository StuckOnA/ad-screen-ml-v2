# =============================================================================
# vision/workers.py
# =============================================================================

import cv2
import queue
import threading

from deepface import DeepFace
from vision.identity import (
    register_insightface_result,
    register_deepface_result,
    register_no_face,
)

MIN_CROP_SIZE = 40
AGE_MIN       = 5
AGE_MAX       = 90

if_queue  = queue.Queue(maxsize=50)
df_queue  = queue.Queue(maxsize=50)
_face_app = None


def setup(app_instance) -> None:
    global _face_app
    _face_app = app_instance


def enqueue_insightface(person_id: int, crop) -> None:
    if not if_queue.full():
        if_queue.put((person_id, crop))


def enqueue_deepface(person_id: int, crop) -> None:
    if not df_queue.full():
        df_queue.put((person_id, crop))


def _is_crop_usable(crop) -> bool:
    if crop is None or crop.size == 0:
        return False
    h, w = crop.shape[:2]
    return h >= MIN_CROP_SIZE and w >= MIN_CROP_SIZE


def _clamp_age(age) -> int | None:
    try:
        age = int(age)
        return age if AGE_MIN <= age <= AGE_MAX else None
    except (TypeError, ValueError):
        return None


def _parse_df_gender(res: dict) -> tuple[str, float]:
    """
    Extract gender and confidence from DeepFace result.
    Uses raw probability scores to avoid Male bias.
    Returns ("M" | "F" | "?", confidence 0.0-1.0).
    """
    gender_data = res.get("gender", {})

    if isinstance(gender_data, dict) and gender_data:
        w_score = float(gender_data.get("Woman", gender_data.get("female", 0)))
        m_score = float(gender_data.get("Man",   gender_data.get("male",   0)))
        gap     = abs(w_score - m_score) / 100.0

        if gap < 0.15:
            return "?", gap
        return ("F" if w_score > m_score else "M"), gap

    dominant = res.get("dominant_gender", "")
    if dominant:
        gender = "F" if str(dominant).lower() in ["woman", "female", "w", "f"] else "M"
        return gender, 0.5

    return "?", 0.0


def insightface_worker() -> None:
    while True:
        person_id, crop = if_queue.get()
        try:
            if not _is_crop_usable(crop):
                register_no_face(person_id)
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            faces    = _face_app.get(crop_rgb)

            if not faces:
                register_no_face(person_id)
                continue

            face      = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            gender    = "M" if face.gender == 1 else "F"
            age       = _clamp_age(face.age)
            det_score = float(face.det_score)

            register_insightface_result(person_id, gender, age, det_score)
            print(f"[IF] ID{person_id} | {gender} ~{age} | det={det_score:.2f}")

        except Exception as e:
            print(f"[IF worker] ID{person_id}: {e}")
            register_no_face(person_id)
        finally:
            if_queue.task_done()


def deepface_worker() -> None:
    while True:
        person_id, crop = df_queue.get()
        try:
            if not _is_crop_usable(crop):
                register_no_face(person_id)
                continue

            results = DeepFace.analyze(
                img_path          = crop,
                actions           = ["age", "gender", "emotion"],
                enforce_detection = False,
                detector_backend  = "retinaface",
                silent            = True,
            )
            res = results[0] if isinstance(results, list) else results

            gender, conf = _parse_df_gender(res)
            age          = _clamp_age(res.get("age"))
            emotion      = res.get("dominant_emotion")
            emo_scores   = res.get("emotion", {})

            register_deepface_result(person_id, gender, age, emotion, emo_scores, conf)
            print(f"[DF] ID{person_id} | {gender} ~{age} | {emotion} | conf={conf:.2f}")

        except Exception as e:
            print(f"[DF worker] ID{person_id}: {e}")
            register_no_face(person_id)
        finally:
            df_queue.task_done()


def start_workers() -> None:
    threading.Thread(target=insightface_worker, daemon=True, name="IF_Worker").start()
    threading.Thread(target=deepface_worker,    daemon=True, name="DF_Worker").start()
    print("Analysis workers started (IF + DF)")