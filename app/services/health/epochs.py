"""
Configuration epochs and change detection -- health spec §2.4; build step 7.

An epoch starts in one of three ways:
  user      the student declares "I changed something" with a change type
  params    a watched parameter differs from the previous flight's
  inferred  three consecutive qualifying flights quarantined for the same
            feature, same direction, consistent with each other: the app asks
            "Did you change something?" (prompt only; a Yes becomes an event)

Resets are per feature, not wholesale: each event resets only the features
its change type affects (the §2.4 table). A feature's baseline group is
therefore set by the most recent event that reset *that feature*. Power
features additionally key off the battery pack (baseline.baseline_group).

Pure functions over a drone's flights in chronological order; the caller
persists events as config_epochs rows (affected_features = reset_features).
"""

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from .baseline import Observation, rolling_baseline
from .features import FEATURE_CATALOGUE

ALL_FEATURES: Set[str] = {k for k, *_ in FEATURE_CATALOGUE}


def _prefixed(*prefixes: str) -> Set[str]:
    return {k for k in ALL_FEATURES if k.startswith(prefixes)}


# Spec §2.4 change type -> features reset.
RESETS: Dict[str, Set[str]] = {
    "props": _prefixed("vib.", "act.") | {"pwr.sag_hover"},
    "motors_escs": _prefixed("vib.", "act.", "trk."),
    "pack": _prefixed("pwr."),
    "frame": ALL_FEATURES - _prefixed("gps."),
    "payload": {"act.hover", "act.sat_frac", "pwr.sag_hover"} | _prefixed("trk."),
    "firmware_tuning": _prefixed("trk.", "vib.band_"),
    "gps": _prefixed("gps.", "est."),
}
CHANGE_TYPES = tuple(RESETS)

# Watched parameters (spec §2.4, PX4 names) -> the change type they signal.
# Rotor geometry is a frame change: moving or re-angling a rotor changes the
# airframe. Firmware is compared as its own pseudo-parameter.
WATCHED: Sequence = (
    (re.compile(r"^MC_(ROLL|PITCH|YAW)RATE_(P|I|D|K)$"), "firmware_tuning"),
    (re.compile(r"^MC_(ROLL|PITCH|YAW)_P$"), "firmware_tuning"),
    (re.compile(r"^IMU_(GYRO|ACCEL|DGYRO)_CUTOFF$"), "firmware_tuning"),
    (re.compile(r"^CA_ROTOR_COUNT$"), "frame"),
    (re.compile(r"^CA_ROTOR(\d+)_(PX|PY|PZ|AX|AY|AZ|KM|CT)$"), "frame"),
    (re.compile(r"^SYS_AUTOSTART$"), "frame"),
    (re.compile(r"^BAT1_(N_CELLS|CAPACITY)$"), "pack"),
)
FIRMWARE_KEY = "firmware_version"
PARAM_REL_TOL = 1e-6   # params are float32 in the log; equal within float noise

# Inferred change (spec §2.4): 3 consecutive quarantined qualifying flights,
# |z| between each pair <= 2.
INFER_FLIGHTS = 3
INFER_MAX_PAIRWISE_Z = 2.0


@dataclass
class ConfigEvent:
    """A change in effect from flight index `flight_index` (0-based) onward."""
    flight_index: int
    change_type: str                 # a CHANGE_TYPES key, or "inferred_confirmed"
    source: str                      # user | params | inferred
    note: str = ""
    features: Optional[Set[str]] = None   # inferred_confirmed: exactly the prompted features
    detail: Dict[str, object] = field(default_factory=dict)

    @property
    def reset_features(self) -> Set[str]:
        if self.features is not None:
            return set(self.features)
        return set(RESETS[self.change_type])


def watched_params(params: Dict[str, float], firmware_version: Optional[str]) -> Dict[str, object]:
    """The watched subset of one log's parameters. Rotor-geometry slots at or
    above CA_ROTOR_COUNT are unused and skipped, so a default in an empty
    slot can't masquerade as a frame change."""
    count = params.get("CA_ROTOR_COUNT")
    out: Dict[str, object] = {}
    for name, value in params.items():
        for pattern, _ in WATCHED:
            m = pattern.match(name)
            if not m:
                continue
            if name.startswith("CA_ROTOR") and m.groups() and count is not None and int(m.group(1)) >= count:
                break
            out[name] = value
            break
    if firmware_version:
        out[FIRMWARE_KEY] = firmware_version
    return out


def _change_type_of(name: str) -> str:
    if name == FIRMWARE_KEY:
        return "firmware_tuning"
    for pattern, change_type in WATCHED:
        if pattern.match(name):
            return change_type
    raise KeyError(name)


def _differs(a, b) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return a != b
    return not math.isclose(float(a), float(b), rel_tol=PARAM_REL_TOL, abs_tol=1e-9)


def param_changes(previous: Dict[str, object], current: Dict[str, object]) -> List[Dict[str, object]]:
    """Watched parameters that changed between two flights. A parameter
    present in one log and absent from the other is not a change: firmware
    versions add and drop parameters, and absence isn't a value."""
    out = []
    for name in sorted(set(previous) & set(current)):
        if _differs(previous[name], current[name]):
            out.append({"param": name, "old": previous[name], "new": current[name],
                        "change_type": _change_type_of(name)})
    return out


def events_from_params(flight_index: int, previous: Dict[str, object],
                       current: Dict[str, object]) -> List[ConfigEvent]:
    """One event per change type detected, all in effect from this flight."""
    by_type: Dict[str, List[dict]] = {}
    for c in param_changes(previous, current):
        by_type.setdefault(c["change_type"], []).append(c)
    return [ConfigEvent(flight_index, t, "params", note=", ".join(c["param"] for c in cs), detail={"changes": cs})
            for t, cs in by_type.items()]


def feature_epoch(feature_key: str, flight_index: int, events: Sequence[ConfigEvent]) -> Optional[int]:
    """Index (into `events`) of the latest event at or before this flight that
    reset this feature; None means the drone's initial epoch."""
    best = None
    for idx, ev in enumerate(events):
        if ev.flight_index <= flight_index and feature_key in ev.reset_features:
            if best is None or ev.flight_index >= events[best].flight_index:
                best = idx
    return best


def latest_epoch(flight_index: int, events: Sequence[ConfigEvent]) -> Optional[int]:
    """The drone-wide epoch (any change type): what flights.epoch_id records."""
    best = None
    for idx, ev in enumerate(events):
        if ev.flight_index <= flight_index and (best is None or ev.flight_index >= events[best].flight_index):
            best = idx
    return best


@dataclass
class ChangePrompt:
    """'Did you change something?' -- raised, never acted on without a Yes."""
    feature: str
    direction: str                   # "up" | "down"
    flight_ids: List[int]
    values: List[float]
    baseline_median: float
    max_pairwise_z: float


def inferred_change(history: Sequence[Observation], feature_key: str) -> Optional[ChangePrompt]:
    """
    Spec §2.4 inferred trigger, over one group's processed history (oldest
    first, admission already decided). Fires when the last INFER_FLIGHTS
    qualifying flights were all quarantined as outliers, all on the same side
    of the rolling baseline they were judged against, and consistent with one
    another (pairwise |difference| / spread <= 2): a new level, not scatter.

    Only z-gate quarantines count. An absolutely-bad quarantine has no
    baseline to have shifted from, and is reported as "consistently bad" by
    the findings instead.
    """
    qualifying = [o for o in history if o.qualified]
    if len(qualifying) < INFER_FLIGHTS:
        return None
    last = qualifying[-INFER_FLIGHTS:]
    if not all(o.admitted is False and (o.quarantine_reason or "").startswith("outlier:") for o in last):
        return None
    # Every one of them was judged against the same frozen rolling baseline:
    # nothing was admitted after the first of them.
    first_pos = list(history).index(last[0])
    prior = [o for o in history[:first_pos] if o.admitted]
    base = rolling_baseline(prior, feature_key)
    if base is None:
        return None
    signs = {1 if o.value > base.median else -1 for o in last}
    if len(signs) != 1:
        return None
    vals = [float(o.value) for o in last]
    pairwise = max(abs(a - b) for i, a in enumerate(vals) for b in vals[i + 1:]) / base.spread
    if pairwise > INFER_MAX_PAIRWISE_Z:
        return None
    return ChangePrompt(feature_key, "up" if signs.pop() > 0 else "down", [o.flight_id for o in last],
                        [round(v, 4) for v in vals], round(base.median, 4), round(pairwise, 2))

