"""
Baseline engine -- health spec §2; build step 5.

Pure functions over a drone's history of one feature, oldest flight first.
No database: the caller loads the rows for one baseline group (an epoch, or a
pack for power features), and this decides which values are admitted, what
the rolling and frozen-reference baselines are, and how far a new value
deviates from them.

  rolling baseline    median and MAD over the last ROLLING_WINDOW admitted
                      values -- catches sudden change
  reference baseline  median and MAD over the first REFERENCE_FLIGHTS admitted
                      values, then frozen -- catches slow drift

Deviation is the robust z, z = (x - median) / max(1.4826 * MAD, floor), where
floor = max(absolute floor, relative floor x |median|),
made one-sided where only one direction is bad.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

MAD_TO_SIGMA = 1.4826  # MAD -> standard deviation for normally distributed data

# TODO(calibrate): window sizes are design choices (spec §2.1, §2.2), not
# fitted. Calibrate with real multi-flight histories: the rolling window
# against how quickly real faults appear, the reference size against how
# stable a newly built drone's first flights actually are.
ROLLING_WINDOW = 10
REFERENCE_FLIGHTS = 5
PERSONAL_MIN_FLIGHTS = 5   # spec §2.2: baseline-scored from 5 admitted flights
BLENDED_MIN_FLIGHTS = 3    # spec §2.2: blended at 3-4

# Iglewicz & Hoaglin (1993), "How to Detect and Handle Outliers": flag
# |modified z| > 3.5. Published for general data; TODO(calibrate) whether
# 3.5 is the right cut for per-flight drone features once real histories
# show their natural spread.
ADMISSION_MAX_MODIFIED_Z = 3.5

HIGHER_IS_WORSE, LOWER_IS_WORSE, TWO_SIDED = "higher_is_worse", "lower_is_worse", "two_sided"

# Minimum spread per feature, in the feature's own units (spec §2.1), so a
# drone with an unusually consistent history doesn't turn a trivial change
# into a huge z. TODO(calibrate): every value below is judgement, not
# calibrated -- revisit once >= 10 real drones each have >= 10 flights,
# setting each floor near the typical flight-to-flight MAD of a healthy drone.
# The first seven are from the spec's table; the rest were not in it and are
# equally uncalibrated.
SIGMA_FLOOR: Dict[str, float] = {
    "vib.metric_p50": 0.5, "vib.metric_p95": 0.5,      # m/s^2 (spec)
    "act.spread": 0.02,                                # fraction (spec)
    "act.hover": 0.02,                                 # 0-1 thrust (spec)
    "pwr.r_cell": 0.5,                                 # mOhm/cell (spec)
    "trk.rate_rms.roll": 0.05, "trk.rate_rms.pitch": 0.05, "trk.rate_rms.yaw": 0.05,  # ratio (spec)
    "gps.eph_p50": 0.3,                                # m (spec)
    "est.test_ratio_p95": 0.1,                         # ratio (spec)
    "vib.clip_rate": 0.5,                              # events/min (not in spec table)
    "vib.band_lo": 1.0, "vib.band_struct": 1.0,        # dB (not in spec table)
    "vib.imbalance": 0.1,                              # PX4 metric units (not in spec table)
    "act.sat_frac": 0.01,                              # fraction (not in spec table)
    "pwr.sag_hover": 0.02,                             # V/cell (not in spec table)
    "trk.att_err_p95": 0.5,                            # deg (not in spec table)
    "gps.nsats_p10": 1.0, "gps.sacc_p50": 0.05,        # count, m/s (not in spec table)
    "gps.noise": 5.0, "gps.jamming": 5.0,              # receiver units (not in spec table)
}

# Relative floor: spread is at least this fraction of the baseline median.
# Added for the vibration metric after the flight-5 sample scored a 29% p95
# rise at 17.3: MAD was 0.18 m/s^2, so the absolute 0.5 floor set the spread,
# and 0.5 is 9% of a 5.4 p50 median but only 4.4% of an 11.3 p95 median --
# p95 was judged twice as strictly, and ArduPilot's VIBE (typically 10-30)
# would be judged far more strictly still. A relative floor scales with the
# drone and the autopilot. TODO(calibrate): 10% is judgement. The one real
# log shows chunk-to-chunk p95 spread of 3-11% and p50 of 27-46% within a
# single flight (driven by hover vs manoeuvring), which bounds nothing about
# whole-flight scatter; set it from the flight-to-flight MAD/median of healthy
# drones once >= 10 drones have >= 10 flights.
RELATIVE_FLOOR: Dict[str, float] = {"vib.metric_p50": 0.10, "vib.metric_p95": 0.10}

# Which direction of change is harmful, per scored feature (spec §2.1).
DIRECTION: Dict[str, str] = {
    "vib.metric_p50": HIGHER_IS_WORSE, "vib.metric_p95": HIGHER_IS_WORSE, "vib.clip_rate": HIGHER_IS_WORSE,
    "vib.band_lo": HIGHER_IS_WORSE, "vib.band_struct": HIGHER_IS_WORSE, "vib.imbalance": HIGHER_IS_WORSE,
    "act.spread": HIGHER_IS_WORSE, "act.sat_frac": HIGHER_IS_WORSE, "act.hover": HIGHER_IS_WORSE,
    "pwr.r_cell": HIGHER_IS_WORSE, "pwr.sag_hover": HIGHER_IS_WORSE,
    "trk.rate_rms.roll": HIGHER_IS_WORSE, "trk.rate_rms.pitch": HIGHER_IS_WORSE,
    "trk.rate_rms.yaw": HIGHER_IS_WORSE, "trk.att_err_p95": HIGHER_IS_WORSE,
    "gps.eph_p50": HIGHER_IS_WORSE, "gps.sacc_p50": HIGHER_IS_WORSE, "gps.noise": HIGHER_IS_WORSE,
    "gps.jamming": HIGHER_IS_WORSE, "est.test_ratio_p95": HIGHER_IS_WORSE,
    "gps.nsats_p10": LOWER_IS_WORSE,
}

# Never baseline-scored: categorical, per-motor detail, or usage-only (§3.4).
NOT_BASELINED = {"act.pattern", "act.mean", "pwr.v_cell_min"}


def baseline_group(feature_key: str, epoch_id: Optional[object], pack_id: Optional[int]) -> str:
    """Spec §2.4. `epoch_id` is the feature's own epoch: the latest change
    that reset this feature (epochs.feature_epoch), None for the drone's
    initial epoch. Power features also keep one baseline per battery pack, and
    a change that resets them (props reset pwr.sag_hover, a frame change all
    of pwr.*) starts a new group within that pack."""
    epoch = "initial" if epoch_id is None else str(epoch_id)
    if feature_key.startswith("pwr."):
        pack = "unknown" if pack_id is None else str(pack_id)
        return f"pack:{pack}" if epoch_id is None else f"pack:{pack}/epoch:{epoch}"
    return f"epoch:{epoch}"


def mode_for(n_admitted: int) -> str:
    """Spec §2.2."""
    if n_admitted >= PERSONAL_MIN_FLIGHTS:
        return "personal"
    if n_admitted >= BLENDED_MIN_FLIGHTS:
        return "blended"
    return "absolute"


@dataclass
class Observation:
    """One flight's value of one feature, plus the facts the gate needs."""
    flight_id: int
    value: Optional[float]
    qualified: bool = True
    valid: bool = True
    absolute_bad: bool = False
    admitted: Optional[bool] = None      # filled in by process_history
    quarantine_reason: Optional[str] = None


@dataclass
class Baseline:
    kind: str                 # "rolling" | "reference"
    median: float
    mad: float
    spread: float             # max(1.4826 * MAD, absolute floor, relative floor * |median|)
    n: int
    flight_ids: List[int] = field(default_factory=list)


def _baseline(kind: str, obs: Sequence[Observation], feature_key: str) -> Optional[Baseline]:
    if not obs:
        return None
    vals = np.array([o.value for o in obs], dtype=np.float64)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    floor = max(SIGMA_FLOOR.get(feature_key, 0.0), RELATIVE_FLOOR.get(feature_key, 0.0) * abs(med))
    return Baseline(kind, med, mad, max(MAD_TO_SIGMA * mad, floor), len(obs), [o.flight_id for o in obs])


def rolling_baseline(admitted: Sequence[Observation], feature_key: str) -> Optional[Baseline]:
    return _baseline("rolling", list(admitted)[-ROLLING_WINDOW:], feature_key)


def reference_baseline(admitted: Sequence[Observation], feature_key: str) -> Optional[Baseline]:
    """Frozen once the epoch has REFERENCE_FLIGHTS admitted values; None before."""
    adm = list(admitted)
    if len(adm) < REFERENCE_FLIGHTS:
        return None
    return _baseline("reference", adm[:REFERENCE_FLIGHTS], feature_key)


def robust_z(value: float, baseline: Baseline, direction: str) -> float:
    """One-sided where only one direction is bad (spec §2.1): the harmless
    direction scores 0, never negative."""
    z = (value - baseline.median) / baseline.spread
    if direction == HIGHER_IS_WORSE:
        return max(0.0, z)
    if direction == LOWER_IS_WORSE:
        return max(0.0, -z)
    return abs(z)


def admission(obs: Observation, prior_admitted: Sequence[Observation], feature_key: str):
    """
    Spec §2.3 gate. Returns (admitted, reason). A value joins the baseline only if:
      (a) the flight qualified, (b) the feature is valid, (c) it isn't
      absolutely bad, and (d) once there are PERSONAL_MIN_FLIGHTS admitted
      values, its modified z against them is within 3.5 -- two-sided here,
      because a big unexplained *improvement* is also a change to ask about
      (spec §2.4), not something to learn silently.
    Rejected values are kept and reported, never learned from.
    """
    if not obs.qualified:
        return False, "flight_not_qualified"
    if not obs.valid or obs.value is None:
        return False, "feature_invalid"
    if obs.absolute_bad:
        return False, "absolute_bad"
    if len(prior_admitted) >= PERSONAL_MIN_FLIGHTS:
        base = rolling_baseline(prior_admitted, feature_key)
        z = robust_z(obs.value, base, TWO_SIDED)
        if z > ADMISSION_MAX_MODIFIED_Z:
            return False, f"outlier:modified_z={z:.2f}"
    return True, None


@dataclass
class HistoryResult:
    observations: List[Observation]
    admitted: List[Observation]
    rolling: Optional[Baseline]
    reference: Optional[Baseline]
    mode: str


def process_history(history: Sequence[Observation], feature_key: str) -> HistoryResult:
    """Walk one group's history oldest-first, deciding admission for each
    value against only the values admitted before it."""
    admitted: List[Observation] = []
    out: List[Observation] = []
    for o in history:
        ok, reason = admission(o, admitted, feature_key)
        o.admitted, o.quarantine_reason = ok, reason
        if ok:
            admitted.append(o)
        out.append(o)
    return HistoryResult(out, admitted, rolling_baseline(admitted, feature_key),
                         reference_baseline(admitted, feature_key), mode_for(len(admitted)))
