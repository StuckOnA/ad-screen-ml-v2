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

# Full-frame queue: (frame_rgb, person_bboxes_dict)
# maxsize=2 prevents frame buildup — drops when worker is busy
if_queue  = queue.Queue(maxsize=2)
df_queue  = queue.Queue(maxsize=50)
_face_app = None


def setup(app_instance) -> None:
    global _face_app
    _face_app = app_instance


def enqueue_insightface_frame(frame, person_bboxes: dict) -> None:
    """
    Enqueue a full frame + person bboxes for InsightFace analysis.
    person_bboxes: {person_id: (x1, y1, x2, y2)}
    """
    if if_queue.full():
        return
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if_queue.put((frame_rgb, person_bboxes))


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


def _match_faces_to_persons(faces, person_bboxes: dict) -> tuple[dict, set]:
    """
    Match InsightFace detected faces to YOLO person bounding boxes.

    For each face, find the person bbox that contains its center.
    If multiple persons contain the same face center, assign to the
    person whose bbox center is closest (Manhattan distance).
    If multiple faces match one person, keep the highest det_score.

    Returns:
        matched:   {person_id: face_object}
        unmatched: set of person_ids with no face
    """
    matched = {}

    for face in faces:
        fx1, fy1, fx2, fy2 = face.bbox[:4]
        face_cx = (fx1 + fx2) / 2
        face_cy = (fy1 + fy2) / 2

        best_pid  = None
        best_dist = float("inf")

        for pid, (px1, py1, px2, py2) in person_bboxes.items():
            if px1 <= face_cx <= px2 and py1 <= face_cy <= py2:
                pcx  = (px1 + px2) / 2
                pcy  = (py1 + py2) / 2
                dist = abs(face_cx - pcx) + abs(face_cy - pcy)
                if dist < best_dist:
                    best_dist = dist
                    best_pid  = pid

        if best_pid is not None:
            if best_pid not in matched or face.det_score > matched[best_pid].det_score:
                matched[best_pid] = face

    unmatched = set(person_bboxes.keys()) - set(matched.keys())
    return matched, unmatched


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
    """
    Full-frame InsightFace analysis.
    Receives (frame_rgb, person_bboxes), detects ALL faces in frame,
    matches them to person bounding boxes, and registers results.
    """
    while True:
        frame_rgb, person_bboxes = if_queue.get()
        try:
            faces = _face_app.get(frame_rgb)

            if not faces:
                for pid in person_bboxes:
                    register_no_face(pid)
                continue

            matched, unmatched = _match_faces_to_persons(faces, person_bboxes)

            for pid, face in matched.items():
                gender    = "M" if face.gender == 1 else "F"
                age       = _clamp_age(face.age)
                det_score = float(face.det_score)
                register_insightface_result(pid, gender, age, det_score)
                print(f"[IF] ID{pid} | {gender} ~{age} | det={det_score:.2f}")

            for pid in unmatched:
                register_no_face(pid)

        except Exception as e:
            print(f"[IF worker] error: {e}")
            for pid in person_bboxes:
                register_no_face(pid)
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

            face_conf = res.get("face_confidence", None)
            if face_conf is None:
                region = res.get("region", {})
                rw, rh = region.get("w", 0), region.get("h", 0)
                ch, cw = crop.shape[:2]
                if rw == 0 or rh == 0 or (rw >= cw * 0.9 and rh >= ch * 0.9):
                    register_no_face(person_id)
                    continue
                face_conf = 0.6
            elif face_conf < 0.5:
                register_no_face(person_id)
                continue

            gender, _  = _parse_df_gender(res)
            age        = _clamp_age(res.get("age"))
            emotion    = res.get("dominant_emotion")
            emo_scores = res.get("emotion", {})

            register_deepface_result(person_id, gender, age, emotion, emo_scores, face_conf)
            print(f"[DF] ID{person_id} | {gender} ~{age} | {emotion} | conf={face_conf:.2f}")

        except Exception as e:
            print(f"[DF worker] ID{person_id}: {e}")
            register_no_face(person_id)
        finally:
            df_queue.task_done()


def start_workers() -> None:
    threading.Thread(target=insightface_worker, daemon=True, name="IF_Worker").start()
    threading.Thread(target=deepface_worker,    daemon=True, name="DF_Worker").start()
    print("Analysis workers started (IF + DF)")
