# =============================================================================
# vision/workers.py — MiVOLO age/gender classification worker
# =============================================================================

import cv2
import queue
import threading
import numpy as np
import torch

from vision.identity import register_mivolo_result, register_no_face

# --- Constants ---
MIN_CROP_SIZE = 40

# --- Queue ---
# (frame_bgr, targets_dict)
# targets_dict: {person_id: {"person_bbox": (x1,y1,x2,y2), "face_bbox": (x1,y1,x2,y2)|None}}
mivolo_queue = queue.Queue(maxsize=2)

# --- Model state ---
_model       = None
_device      = None
_meta        = None
_input_size  = None
_data_config = None


def setup(mivolo_checkpoint: str, device: str) -> None:
    """Load MiVOLO age/gender model."""
    global _model, _device, _meta, _input_size, _data_config

    from mivolo.model.mi_volo import MiVOLO, Meta

    _device = torch.device(device)

    _meta = Meta().load_from_ckpt(mivolo_checkpoint, disable_faces=False, use_persons=True)
    print(f"[MiVOLO] Meta: min_age={_meta.min_age}, max_age={_meta.max_age}, "
          f"avg_age={_meta.avg_age}, input_size={_meta.input_size}")

    _model = MiVOLO(
        ckpt_path=mivolo_checkpoint,
        device=device,
        half=(_device.type != "cpu"),
        use_persons=True,
        disable_faces=False,
        verbose=False,
    )
    _input_size  = _model.input_size
    _data_config = _model.data_config
    print(f"[MiVOLO] Model loaded. Input size: {_input_size}, device: {device}")


def enqueue_mivolo(frame, targets: dict) -> bool:
    """
    Enqueue a frame + analysis targets for MiVOLO classification.
    targets: {person_id: {"person_bbox": (x1,y1,x2,y2), "face_bbox": (x1,y1,x2,y2)|None}}
    Returns True if enqueued, False if queue was full.
    """
    if mivolo_queue.full():
        return False
    mivolo_queue.put((frame, targets))
    return True


def _letterbox(im: np.ndarray, new_shape: int) -> np.ndarray:
    """Resize + pad to square while preserving aspect ratio."""
    h, w = im.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_h, new_w = int(round(h * r)), int(round(w * r))
    if (h, w) != (new_h, new_w):
        im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dh = new_shape - new_h
    dw = new_shape - new_w
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return im


def _preprocess_crop(crop: np.ndarray) -> torch.Tensor:
    """BGR crop → normalized tensor [1, 3, H, W]."""
    img = _letterbox(crop, _input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    mean = np.array(_data_config["mean"], dtype=np.float32)
    std  = np.array(_data_config["std"], dtype=np.float32)
    img  = (img - mean) / std
    img  = img.transpose((2, 0, 1))  # HWC → CHW
    return torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0)


def _make_zero_tensor() -> torch.Tensor:
    """Create a zero-filled 3-channel tensor (used when face/body missing)."""
    mean = torch.tensor(_data_config["mean"], dtype=torch.float32)
    std  = torch.tensor(_data_config["std"], dtype=torch.float32)
    img  = torch.zeros(3, _input_size, _input_size, dtype=torch.float32)
    # Normalize zeros: (0 - mean) / std
    for c in range(3):
        img[c] = (img[c] - mean[c]) / std[c]
    return img.unsqueeze(0)


def _crop_bbox(frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
    """Crop a bounding box from frame, returns None if too small."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.shape[0] < MIN_CROP_SIZE or crop.shape[1] < MIN_CROP_SIZE:
        return None
    return crop.copy()


def _decode_output(output: torch.Tensor, index: int) -> tuple:
    """
    Decode MiVOLO model output for one person.
    Returns: (gender_str, age_int, gender_score)
    """
    age_raw = output[index, 2].item()
    age = age_raw * (_meta.max_age - _meta.min_age) + _meta.avg_age
    age = max(1, min(100, round(age)))

    gender_logits = output[index, :2].softmax(-1)
    gender_score  = gender_logits.max().item()
    gender_idx    = gender_logits.argmax().item()
    # MiVOLO: index 0 = male, index 1 = female
    gender = "M" if gender_idx == 0 else "F"

    return gender, age, gender_score


def mivolo_worker() -> None:
    """
    Background worker: receives (frame, targets), runs MiVOLO inference,
    registers results to identity memory.
    """
    while True:
        frame, targets = mivolo_queue.get()
        try:
            if not targets:
                continue

            # Prepare batch
            face_tensors = []
            body_tensors = []
            person_ids   = []

            zero_tensor = _make_zero_tensor()

            for pid, info in targets.items():
                person_bbox = info["person_bbox"]
                face_bbox   = info["face_bbox"]

                # Body crop
                body_crop = _crop_bbox(frame, person_bbox)
                if body_crop is not None:
                    body_tensor = _preprocess_crop(body_crop)
                else:
                    body_tensor = zero_tensor.clone()

                # Face crop
                if face_bbox is not None:
                    face_crop = _crop_bbox(frame, face_bbox)
                    if face_crop is not None:
                        face_tensor = _preprocess_crop(face_crop)
                    else:
                        face_tensor = zero_tensor.clone()
                        # Face bbox existed but crop too small → still count as face found
                else:
                    face_tensor = zero_tensor.clone()

                face_tensors.append(face_tensor)
                body_tensors.append(body_tensor)
                person_ids.append(pid)

            if not person_ids:
                continue

            # Build 6-channel input: [face_channels, body_channels]
            faces_batch  = torch.cat(face_tensors, dim=0).to(_device)
            bodies_batch = torch.cat(body_tensors, dim=0).to(_device)
            model_input  = torch.cat((faces_batch, bodies_batch), dim=1)

            # Inference
            output = _model.inference(model_input)

            # Decode and register results
            for i, pid in enumerate(person_ids):
                gender, age, gender_score = _decode_output(output, i)
                face_found = targets[pid]["face_bbox"] is not None

                register_mivolo_result(pid, gender, age, gender_score, face_found)

                if not face_found:
                    register_no_face(pid)

                tag = "face+body" if face_found else "body-only"
                print(f"[MV] ID{pid} | {gender} ~{age} | "
                      f"score={gender_score:.2f} | {tag}")

        except Exception as e:
            print(f"[MiVOLO worker] error: {e}")
            for pid in targets:
                register_no_face(pid)
        finally:
            mivolo_queue.task_done()


def start_workers() -> None:
    threading.Thread(target=mivolo_worker, daemon=True, name="MiVOLO_Worker").start()
    print("[MiVOLO] Worker started")
