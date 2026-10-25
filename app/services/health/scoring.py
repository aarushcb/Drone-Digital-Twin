"""
Scoring -- health spec §3.3, §3.4, §4.1, §4.2; build step 8.

Per feature, with n = qualifying flights in the feature's baseline group
*including this one* and the baseline built from the admitted flights *before*
it (spec §4.1, "Counting n"):

  S_baseline = clamp(100 - 15 * (z+ - 1), 0, 100)       z+ one-sided robust z
  S_absolute = norms.absolute_score                     sliding / pass-fail / trigger-only
  S          = w * S_baseline + (1 - w) * S_absolute,   w = 0 (n <= 2), 1/3, 2/3, 1 (n >= 5)

No absolute norm: unscored ("learning") until n = 3, then S_baseline alone.
An absolute bad caps S at 49 at every n, forever.

Subsystem and overall: 0.5 * worst + 0.5 * mean over what is scored; the
overall is null unless at least 4 of the 5 subsystems are scored.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from . import baseline as B
from . import norms as N
from .features import ATT, MOT, POS, PWR, VIB, Feature

SUBSYSTEMS = (ATT, POS, VIB, MOT, PWR)
DISPLAY_SUBSYSTEM = {ATT: "Attitude & control", POS: "Position", VIB: "Vibration", MOT: "Motors", PWR: "Power"}
OVERALL_MIN_SUBSYSTEMS = 4      # spec §4.2
ABSOLUTE_BAD_CAP = 49.0         # spec §3.3
SCORED_FROM_N = 3               # spec §3.3: baseline-only features score from n = 3
CONFIDENCE_HIGH_N = 10          # spec §5.2
EST_REJECTION_RATIO = 1.0       # PX4 rejects a measurement at test ratio 1 (spec §3.2)


def blend_weight(n: int) -> float:
    """Spec §3.3: w = 0 for n <= 2, (n - 2) / 3 for n = 3-4, 1 for n >= 5."""
    if n <= 2:
        return 0.0
    if n >= B.PERSONAL_MIN_FLIGHTS:
        return 1.0
    return (n - 2) / 3.0


def baseline_score(z_plus: float) -> float:
    """Spec §4.1: z <= 1 -> 100, z = 2 -> 85, z = 3 -> 70, z ~ 4.3 -> 50."""
    return max(0.0, min(100.0, 100.0 - 15.0 * (z_plus - 1.0)))


def combine(scores: Sequence[float]) -> Optional[float]:
    """Spec §4.2: 0.5 * min + 0.5 * mean."""
    if not scores:
        return None
    return 0.5 * min(scores) + 0.5 * sum(scores) / len(scores)


def definitional(feature: Feature) -> Optional[str]:
    """Spec §3.2 / §3.4: facts that raise a finding from flight one without
    making their subsystem scored."""
    if feature.value is None:
        return None
    if feature.key == "act.sat_frac" and feature.value > 0:
        return "motor_saturation"
    if feature.key == "est.test_ratio_p95" and feature.value >= EST_REJECTION_RATIO:
        return "estimator_rejection"
    return None


@dataclass
class FeatureScore:
    feature: Feature
    group: str
    n: int                                  # qualifying flights in group incl. this one (admitted before + 1)
    flights_in_group: int                   # every qualifying flight with this feature valid, incl. this one
    mode: Optional[str]
    weight: float
    absolute: N.AbsoluteResult
    rolling: Optional[B.Baseline] = None    # from flights before this one
    reference: Optional[B.Baseline] = None
    z: Optional[float] = None
    s_baseline: Optional[float] = None
    score: Optional[float] = None
    capped: bool = False
    status: str = "unscored"                # scored | learning | usage | unscored
    admitted: Optional[bool] = None
    quarantine_reason: Optional[str] = None
    consistently_bad: bool = False
    definitional: Optional[str] = None
    confidence: Optional[str] = None
    basis: Optional[str] = None
    unscored_reason: Optional[str] = None

    @property
    def key(self) -> str:
        return self.feature.key

    @property
    def absolute_bad(self) -> bool:
        return self.absolute.score is not None and self.absolute.score < 50.0


def _guide_owner(norm: N.Norm) -> str:
    return "ArduPilot" if "ardupilot.org" in norm.url else "PX4"


def confidence_for(fs: "FeatureScore") -> str:
    """Spec §5.2."""
    tier = fs.feature.tier
    if fs.mode == "personal":
        conf = "high" if fs.n >= CONFIDENCE_HIGH_N and tier == "R1" else "medium"
    elif fs.mode == "blended":
        conf = "medium"
    elif fs.absolute.norm is not None and fs.absolute.score is not None:
        conf = fs.absolute.norm.confidence
    elif fs.definitional:
        conf = "high"           # definition / PX4's own rejection level (spec §3.4 checklist)
    else:
        conf = "low"
    if tier == "R2" and fs.n < B.PERSONAL_MIN_FLIGHTS:
        conf = "low"
    return conf


def score_feature(feature: Feature, autopilot: str, group: str, prior: Sequence[B.Observation],
                  current: B.Observation) -> FeatureScore:
    """
    `prior` is the group's processed history before this flight (admission
    already decided); `current` is this flight's observation, already run
    through the admission gate against it.
    """
    absolute = N.absolute_score(feature.key, feature.value, autopilot, feature.detail)
    prior_admitted = [o for o in prior if o.admitted]
    prior_valid = [o for o in prior if o.qualified and o.valid and o.value is not None]
    n = len(prior_admitted) + 1
    fs = FeatureScore(feature, group, n, len(prior_valid) + 1, B.mode_for(n), blend_weight(n), absolute,
                      admitted=current.admitted, quarantine_reason=current.quarantine_reason,
                      definitional=definitional(feature))

    # Never-scored features carry no mode: there is no comparison behind them.
    if not feature.valid:
        fs.unscored_reason, fs.mode = feature.reason, None
        return fs
    if feature.tier == "R3":
        fs.unscored_reason, fs.mode = "tier_r3_never_scored", None
        return fs
    if absolute.status.startswith("usage"):
        fs.status, fs.unscored_reason, fs.mode = "usage", "usage_note_only", None
        fs.confidence = absolute.norm.confidence
        return fs
    if feature.key in B.NOT_BASELINED or feature.value is None:
        fs.unscored_reason, fs.mode = "detail_only", None
        return fs

    fs.rolling = B.rolling_baseline(prior_admitted, feature.key)
    fs.reference = B.reference_baseline(prior_admitted, feature.key)
    if fs.rolling is not None:
        fs.z = B.robust_z(feature.value, fs.rolling, B.DIRECTION[feature.key])
        fs.s_baseline = baseline_score(fs.z)

    s_abs = absolute.score
    use_baseline = fs.s_baseline is not None and n >= SCORED_FROM_N
    if s_abs is None:
        fs.score = fs.s_baseline if use_baseline else None
        if use_baseline:
            fs.weight = 1.0
    elif use_baseline:
        fs.score = fs.weight * fs.s_baseline + (1.0 - fs.weight) * s_abs
    else:
        fs.score, fs.weight = s_abs, 0.0
    if fs.absolute_bad and fs.score is not None and fs.score > ABSOLUTE_BAD_CAP:
        fs.score, fs.capped = ABSOLUTE_BAD_CAP, True
    if fs.score is not None:
        fs.score = round(fs.score, 1)

    # Mode is the §2.2 flight-count mode, except that a score with no
    # baseline behind it is absolute whatever n says.
    fs.status = "learning" if fs.score is None else "scored"
    fs.mode = B.mode_for(n) if use_baseline else "absolute"
    fs.consistently_bad = fs.absolute_bad and fs.flights_in_group >= 2 and all(o.absolute_bad for o in prior_valid)

    parts = []
    if fs.weight < 1 and absolute.norm is not None and s_abs is not None:
        parts.append(f"vs {_guide_owner(absolute.norm)} guide")
    if fs.weight > 0 and fs.rolling is not None:
        parts.append(f"vs your last {fs.rolling.n} flight{'s' if fs.rolling.n != 1 else ''}")
    fs.basis = " + ".join(parts) or None
    fs.confidence = confidence_for(fs)
    return fs


@dataclass
class Rollup:
    subsystem_scores: Dict[str, Optional[float]]
    overall: Optional[float]
    score_band: Optional[str]
    scored_subsystems: int
    coverage: Dict[str, int] = field(default_factory=dict)


def rollup(scores: Sequence[FeatureScore]) -> Rollup:
    subs: Dict[str, Optional[float]] = {}
    for sub in SUBSYSTEMS:
        vals = [fs.score for fs in scores if fs.feature.subsystem == sub and fs.status == "scored"]
        c = combine(vals)
        subs[sub] = None if c is None else round(c, 1)
    scored = [v for v in subs.values() if v is not None]
    overall = combine(scored) if len(scored) >= OVERALL_MIN_SUBSYSTEMS else None
    overall = None if overall is None else round(overall, 1)
    coverage = {"features_scored": sum(fs.status == "scored" for fs in scores),
                "features_valid": sum(fs.feature.valid for fs in scores),
                "features_total": len(scores)}
    return Rollup(subs, overall, None if overall is None else N.band_status(overall), len(scored), coverage)


# ---------------------------------------------------------------- flight-one checklist (spec §3.4)

def _fmt(v: float, d: int) -> str:
    return f"{v:.{d}f}"


def build_checks(scores: Dict[str, FeatureScore], autopilot: str) -> List[dict]:
    """The cold-start checklist. Rows whose feature couldn't be computed are
    'unavailable' and don't count toward 'k of m checks OK'."""
    px4 = autopilot == "px4"

    def row(key, label, feature_key, figure_fn, passes_fn, basis, confidence, fail_status=None):
        fs = scores.get(feature_key)
        if fs is None or not fs.feature.valid or fs.feature.value is None:
            return {"key": key, "label": label, "status": "unavailable", "figure": None, "basis": basis,
                    "confidence": confidence, "reason": None if fs is None else fs.feature.reason}
        v = fs.feature.value
        ok = passes_fn(v, fs)
        if ok:
            status = "ok"
        elif fail_status is not None:
            status = fail_status
        else:
            status = fs.absolute.status if fs.absolute.score is not None else "watch"
        return {"key": key, "label": label, "status": status, "figure": figure_fn(v, fs), "basis": basis,
                "confidence": confidence}

    vib_edge = 4.905 if px4 else 30.0
    rows = [
        row("vibration_level", "Vibration level", "vib.metric_p50", lambda v, f: f"{_fmt(v, 1)} m/s²",
            lambda v, f: v < vib_edge, f"{'PX4 Flight Review' if px4 else 'ArduPilot'} vibration guide",
            "low" if px4 else "medium"),
        row("accel_clipping", "Accelerometer clipping", "vib.clip_rate",
            lambda v, f: f"{int(f.feature.detail.get('clip_events', 0))} events",
            lambda v, f: int(f.feature.detail.get("clip_events", 0)) == 0, "ArduPilot clipping guidance", "low"),
    ]
    if px4:
        rows += [
            row("prop_imbalance", "Prop imbalance (PX4 detector)", "vib.imbalance", lambda v, f: _fmt(v, 2),
                lambda v, f: v < 30.0, "PX4 FD_IMB_PROP_THR", "medium"),
            row("gps_accuracy", "GPS accuracy", "gps.eph_p50", lambda v, f: f"{_fmt(v, 2)} m",
                lambda v, f: v <= 3.0, "PX4 EKF2_REQ_EPH", "medium"),
            row("gps_satellites", "GPS satellites", "gps.nsats_p10", lambda v, f: f"{int(round(v))}",
                lambda v, f: v >= 6, "PX4 EKF2_REQ_NSATS", "medium"),
            row("gps_speed_accuracy", "GPS speed accuracy", "gps.sacc_p50", lambda v, f: f"{_fmt(v, 2)} m/s",
                lambda v, f: v <= 0.5, "PX4 EKF2_REQ_SACC", "medium"),
        ]
    rows += [
        row("estimator_agreement", "Estimator agreement", "est.test_ratio_p95", lambda v, f: _fmt(v, 2),
            lambda v, f: v < EST_REJECTION_RATIO, "PX4 rejection level (test ratio 1)", "high", fail_status="off"),
        row("motor_headroom", "Motor headroom", "act.sat_frac", lambda v, f: f"{_fmt(100 * v, 1)}% of flight",
            lambda v, f: v == 0, "definition: a motor at full output has no margin", "high", fail_status="watch"),
        row("pack_not_empty", "Pack not flown to empty", "pwr.v_cell_min", lambda v, f: f"{_fmt(v, 2)} V/cell",
            lambda v, f: v > 3.6, "PX4 BAT_V_EMPTY (usage note, not health)", "high", fail_status="usage_note"),
    ]
    return rows


# ---------------------------------------------------------------- logging gaps (spec §3.4 item 7)

# Feature null reason -> what would let the next log support it.
# SDLOG_PROFILE bit 9 is "raw FIFO high-rate IMU (accel)" in PX4 v1.14's
# logger parameters, as the spec states; re-verify against the firmware in use.
LOGGING_GAPS = {
    "requires_fifo_accel": ("Rotor-band vibration (above the accel filter cutoff)",
                            "Set SDLOG_PROFILE bit 9 (raw FIFO high-rate accel)."),
    "no_per_cell_sensing": ("Per-cell voltage spread",
                            "Needs a battery monitor with per-cell sensing (smart pack or balance lead); "
                            "no logging setting adds it."),
}


def logging_gaps(features: Sequence[Feature]) -> List[dict]:
    out = []
    for f in features:
        if f.reason in LOGGING_GAPS:
            what, how = LOGGING_GAPS[f.reason]
            out.append({"feature": f.key, "reason": f.reason, "what": what, "how": how})
    return out
