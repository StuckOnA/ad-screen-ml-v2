<<<<<<< HEAD
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
    PRECISION_FRAMES_REQUIRED,
    PRECISION_CONF_THRESHOLD,
    PRECISION_BBOX_AREA,
    REANALYZE_BUCKETS,
    REANALYZE_NONE,
    BUCKET_HIGH_THRESHOLD,
    BUCKET_MEDIUM_THRESHOLD,
    FACING_AWAY_MISS_THRESHOLD,
    FACING_AWAY_INITIAL_THRESHOLD,
)

MAX_HISTORY = 5

identity_memory: dict = {}
memory_lock = threading.Lock()


def new_entry(conf: float, now: float) -> dict:
    return {
        "last_seen":          now,
        "last_analyzed":      0.0,
        "frames_seen":        1,
        "avg_conf":           conf,
        "facing_away":        False,
        "consecutive_misses": 0,
        "age":                None,
        "gender":             "?",
        "emotion":            None,
        "source":             None,
        "result_confidence":  None,
        "history_age":        [],
        "history_gender":     [],
        "history_emotion":    [],
        "bbox_area":          0,
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


def get_analysis_tier(mem: dict, bbox_area: int) -> str:
    if (
        mem["frames_seen"] >= PRECISION_FRAMES_REQUIRED and
        mem["avg_conf"]    >= PRECISION_CONF_THRESHOLD  and
        bbox_area          >= PRECISION_BBOX_AREA
    ):
        return "precision"
    elif is_stable(mem):
        return "standard"
    else:
        return "noisy"


def get_reanalyze_interval(mem: dict) -> float:
    score = mem.get("result_confidence", None)
    if score is None:
        return REANALYZE_NONE
    elif score >= BUCKET_HIGH_THRESHOLD:
        return REANALYZE_BUCKETS["high"]
    elif score >= BUCKET_MEDIUM_THRESHOLD:
        return REANALYZE_BUCKETS["medium"]
    else:
        return REANALYZE_BUCKETS["low"]


def bucket_label(mem: dict) -> str:
    score = mem.get("result_confidence", None)
    if score is None:                      return "?"
    elif score >= BUCKET_HIGH_THRESHOLD:   return "H"
    elif score >= BUCKET_MEDIUM_THRESHOLD: return "M"
    else:                                  return "L"


def register_insightface_result(person_id: int, gender: str,
                                 age: int, det_score: float) -> None:
    with memory_lock:
        if person_id not in identity_memory:
            return
        mem = identity_memory[person_id]
        mem["last_analyzed"]     = time.time()
        mem["source"]            = "IF"
        mem["result_confidence"] = det_score
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


def register_deepface_result(person_id: int, gender: str, age: int,
                              emotion: str, emo_scores: dict,
                              conf: float) -> None:
    with memory_lock:
        if person_id not in identity_memory:
            return
        mem = identity_memory[person_id]
        mem["last_analyzed"]      = time.time()
        mem["source"]             = "DF"
        mem["result_confidence"]  = conf
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

        if emotion:
            mem["history_emotion"].append(emotion)
            if len(mem["history_emotion"]) > MAX_HISTORY:
                mem["history_emotion"].pop(0)
            mem["emotion"] = Counter(mem["history_emotion"]).most_common(1)[0][0]


def register_no_face(person_id: int) -> None:
    """
    Called when worker finds no face in crop.
    Increments miss counter — if misses exceed threshold, marks facing_away.
    facing_away flips back to False the moment a face is found again.
    """
    with memory_lock:
        if person_id not in identity_memory:
            return
        mem           = identity_memory[person_id]
        never_labeled = mem["gender"] == "?" and mem["age"] is None
        threshold     = (FACING_AWAY_INITIAL_THRESHOLD if never_labeled
                         else FACING_AWAY_MISS_THRESHOLD)

        mem["last_analyzed"]      = time.time()
        mem["result_confidence"]  = 0.0
        mem["consecutive_misses"] += 1

        if mem["consecutive_misses"] >= threshold:
            mem["facing_away"] = True


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
=======
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
    PRECISION_FRAMES_REQUIRED,
    PRECISION_CONF_THRESHOLD,
    PRECISION_BBOX_AREA,
    REANALYZE_BUCKETS,
    REANALYZE_NONE,
    BUCKET_HIGH_THRESHOLD,
    BUCKET_MEDIUM_THRESHOLD,
    FACING_AWAY_MISS_THRESHOLD,
    FACING_AWAY_INITIAL_THRESHOLD,
)

MAX_HISTORY = 5

identity_memory: dict = {}
memory_lock = threading.Lock()


def new_entry(conf: float, now: float) -> dict:
    return {
        "last_seen":          now,
        "last_analyzed":      0.0,
        "frames_seen":        1,
        "avg_conf":           conf,
        "facing_away":        False,
        "consecutive_misses": 0,
        "age":                None,
        "gender":             "?",
        "emotion":            None,
        "source":             None,
        "result_confidence":  None,
        "history_age":        [],
        "history_gender":     [],
        "history_emotion":    [],
        "bbox_area":          0,
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


def get_analysis_tier(mem: dict, bbox_area: int) -> str:
    if (
        mem["frames_seen"] >= PRECISION_FRAMES_REQUIRED and
        mem["avg_conf"]    >= PRECISION_CONF_THRESHOLD  and
        bbox_area          >= PRECISION_BBOX_AREA
    ):
        return "precision"
    elif is_stable(mem):
        return "standard"
    else:
        return "noisy"


def get_reanalyze_interval(mem: dict) -> float:
    score = mem.get("result_confidence", None)
    if score is None:
        return REANALYZE_NONE
    elif score >= BUCKET_HIGH_THRESHOLD:
        return REANALYZE_BUCKETS["high"]
    elif score >= BUCKET_MEDIUM_THRESHOLD:
        return REANALYZE_BUCKETS["medium"]
    else:
        return REANALYZE_BUCKETS["low"]


def bucket_label(mem: dict) -> str:
    score = mem.get("result_confidence", None)
    if score is None:                      return "?"
    elif score >= BUCKET_HIGH_THRESHOLD:   return "H"
    elif score >= BUCKET_MEDIUM_THRESHOLD: return "M"
    else:                                  return "L"


def register_insightface_result(person_id: int, gender: str,
                                 age: int, det_score: float) -> None:
    with memory_lock:
        if person_id not in identity_memory:
            return
        mem = identity_memory[person_id]
        mem["last_analyzed"]     = time.time()
        mem["source"]            = "IF"
        mem["result_confidence"] = det_score
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


def register_deepface_result(person_id: int, gender: str, age: int,
                              emotion: str, emo_scores: dict,
                              conf: float) -> None:
    with memory_lock:
        if person_id not in identity_memory:
            return
        mem = identity_memory[person_id]
        mem["last_analyzed"]      = time.time()
        mem["source"]             = "DF"
        mem["result_confidence"]  = conf
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

        if emotion:
            mem["history_emotion"].append(emotion)
            if len(mem["history_emotion"]) > MAX_HISTORY:
                mem["history_emotion"].pop(0)
            mem["emotion"] = Counter(mem["history_emotion"]).most_common(1)[0][0]


def register_no_face(person_id: int) -> None:
    """
    Called when worker finds no face in crop.
    Increments miss counter — if misses exceed threshold, marks facing_away.
    facing_away flips back to False the moment a face is found again.
    """
    with memory_lock:
        if person_id not in identity_memory:
            return
        mem           = identity_memory[person_id]
        never_labeled = mem["gender"] == "?" and mem["age"] is None
        threshold     = (FACING_AWAY_INITIAL_THRESHOLD if never_labeled
                         else FACING_AWAY_MISS_THRESHOLD)

        mem["last_analyzed"]      = time.time()
        mem["result_confidence"]  = 0.0
        mem["consecutive_misses"] += 1

        if mem["consecutive_misses"] >= threshold:
            mem["facing_away"] = True


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
>>>>>>> fbc40b2 (Initial Commit)
            del identity_memory[pid]