# =============================================================================
# vision/identity.py
# =============================================================================

import time
import threading
import statistics
import copy
from collections import Counter

from config import (
    STALE_TIMEOUT,
    STABLE_FRAMES_REQUIRED,
    STABLE_CONF_THRESHOLD,
    REANALYZE_BUCKETS,
    REANALYZE_NONE,
    AGREEMENT_HIGH_THRESHOLD,
    AGREEMENT_LOW_THRESHOLD,
    FACING_AWAY_MISS_THRESHOLD,
    FACING_AWAY_INITIAL_THRESHOLD,
)

MAX_HISTORY = 5

identity_memory: dict = {}
memory_lock = threading.Lock()


def new_entry(conf: float, now: float, bbox_area: int = 0) -> dict:
    return {
        "last_seen":          now,
        "last_analyzed":      0.0,
        "frames_seen":        1,
        "avg_conf":           conf,
        "facing_away":        False,
        "consecutive_misses": 0,
        "age":                None,
        "gender":             "?",
        "gender_score":       0.0,
        "source":             None,
        "history_age":        [],
        "history_gender":     [],
        "bbox_area":          bbox_area,
        "area_history":       [],
        "movement":           "stable",
    }


def update_entry(mem: dict, conf: float, now: float, bbox_area: int) -> None:
    mem["last_seen"]    = now
    mem["frames_seen"] += 1
    mem["avg_conf"]     = mem["avg_conf"] * 0.9 + conf * 0.1
    mem["bbox_area"]    = bbox_area


def is_stable(mem: dict) -> bool:
    return (
        mem["frames_seen"] >= STABLE_FRAMES_REQUIRED and
        mem["avg_conf"]    >= STABLE_CONF_THRESHOLD
    )


def _compute_agreement(mem: dict) -> float:
    """
    Compute agreement score from gender history.
    Requires minimum 3 readings before trusting agreement.
    """
    history = mem.get("history_gender", [])
    if len(history) < 3:
        return 0.0
    most_common_count = Counter(history).most_common(1)[0][1]
    return most_common_count / len(history)


def get_reanalyze_interval(mem: dict) -> float:
    agreement = _compute_agreement(mem)
    if agreement == 0.0:
        return REANALYZE_NONE
    elif agreement >= AGREEMENT_HIGH_THRESHOLD:
        return REANALYZE_BUCKETS["high"]
    elif agreement >= AGREEMENT_LOW_THRESHOLD:
        return REANALYZE_BUCKETS["medium"]
    else:
        return REANALYZE_BUCKETS["low"]


def bucket_label(mem: dict) -> str:
    agreement = _compute_agreement(mem)
    if agreement == 0.0:                        return "?"
    elif agreement >= AGREEMENT_HIGH_THRESHOLD: return "H"
    elif agreement >= AGREEMENT_LOW_THRESHOLD:  return "M"
    else:                                       return "L"


def register_mivolo_result(person_id: int, gender: str, age: int,
                            gender_score: float, face_found: bool = True) -> None:
    """Register MiVOLO classification result (gender + age)."""
    with memory_lock:
        if person_id not in identity_memory:
            return
        mem = identity_memory[person_id]
        mem["last_analyzed"]  = time.time()
        mem["source"]         = "MV"
        mem["gender_score"]   = gender_score

        # Only clear facing_away when face is actually visible
        if face_found:
            mem["consecutive_misses"] = 0
            mem["facing_away"]        = False

        if gender and gender != "?":
            mem["history_gender"].append(gender)
            if len(mem["history_gender"]) > MAX_HISTORY:
                mem["history_gender"].pop(0)
            mem["gender"] = Counter(mem["history_gender"]).most_common(1)[0][0]

        if age is not None:
            mem["history_age"].append(age)
            if len(mem["history_age"]) > MAX_HISTORY:
                mem["history_age"].pop(0)
            mem["age"] = int(statistics.median(mem["history_age"]))


def register_no_face(person_id: int) -> None:
    """
    Called when no face is detected for a person.
    Increments miss counter — if misses exceed threshold, marks facing_away.
    """
    with memory_lock:
        if person_id not in identity_memory:
            return
        mem           = identity_memory[person_id]
        never_labeled = mem["gender"] == "?" and mem["age"] is None
        threshold     = (FACING_AWAY_INITIAL_THRESHOLD if never_labeled
                         else FACING_AWAY_MISS_THRESHOLD)

        mem["consecutive_misses"] += 1

        if mem["consecutive_misses"] >= threshold:
            mem["facing_away"] = True


def prune_stale_identities() -> None:
    now = time.time()
    with memory_lock:
        stale = [
            pid for pid, mem in identity_memory.items()
            if (now - mem.get("last_seen", 0)) > STALE_TIMEOUT
        ]
        for pid in stale:
            del identity_memory[pid]


def get_snapshot() -> dict:
    with memory_lock:
        return copy.deepcopy(identity_memory)


def get_entry_snapshot(person_id: int) -> dict:
    with memory_lock:
        mem = identity_memory.get(person_id)
        return copy.deepcopy(mem) if mem else {}


def prune_stale_identities() -> None:
    now = time.time()
    with memory_lock:
        stale = [
            pid for pid, mem in identity_memory.items()
            if (now - mem.get("last_seen", 0)) > STALE_TIMEOUT
        ]
        for pid in stale:
            del identity_memory[pid]
