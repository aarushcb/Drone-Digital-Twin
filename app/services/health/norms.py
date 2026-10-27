"""
Published thresholds and absolute (cold-start) scoring -- health spec §3.1,
§3.4, §4.1; build step 4.

Every threshold is a named Norm carrying its source, the URL, the date it was
checked against that source, and its confidence per spec §5.2. Nothing here
is a number we chose; where a published source gives only one edge, the
scoring says so (GATE) rather than inventing a second one.

Kinds of norm:
  GRADED    healthy edge and bad edge both published. Piecewise-linear:
            0 -> 100, healthy edge -> 85, bad edge -> 50, 2 x bad edge -> 0.
  GATE      one published pass/fail edge (PX4's EKF GPS requirements).
            Pass -> 100, fail -> GATE_FAIL_SCORE. No graded severity, because
            no source grades it.
  BAD_ONLY  only a failure trigger is published (PX4's imbalance detector).
            At or past it -> BAD_ONLY_SCORE; below it -> no absolute score
            (baseline-only).
  USAGE     never scored at any flight count (spec §3.4): a usage note only.
            Min cell voltage vs PX4's *empty* level measures how far the pack
            was flown, not whether it is healthy.
  CONTEXT   never scored, never "absolutely bad", never a Trends guide line:
            the value is reported with where a published line sits (spec
            §3.1, amended 2026-10-27). The vibration level, whose lines judged
            real drones wrongly more often than not.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

CHECKED = "2026-10-25"

GRADED, GATE, BAD_ONLY, USAGE, CONTEXT = "graded", "gate", "bad_only", "usage", "context"
HIGHER_IS_WORSE, LOWER_IS_WORSE = "higher_is_worse", "lower_is_worse"

# TODO(calibrate): a failed single-edge gate has no published severity; 70
# (bottom of the "watch" band) is a judgement. Calibrate against how often
# real flights failing EKF2_REQ_* actually show position problems.
GATE_FAIL_SCORE = 70.0
# TODO(calibrate): PX4 publishes only the imbalance *failure* trigger; 40
# (inside the "bad" band) is a judgement for "PX4 itself would fail this".
BAD_ONLY_SCORE = 40.0

PX4_FLIGHT_REVIEW_PLOTS = "https://github.com/PX4/flight_review/blob/main/app/plot_app/configured_plots.py"
AP_VIBRATION = "https://ardupilot.org/copter/docs/common-measuring-vibration.html"
PX4_EKF2_PARAMS_114 = "https://github.com/PX4/PX4-Autopilot/blob/release/1.14/src/modules/ekf2/ekf2_params.c"
PX4_BATTERY_PARAMS_114 = "https://github.com/PX4/PX4-Autopilot/blob/release/1.14/src/lib/battery/module.yaml"
PX4_FD_PARAMS_114 = ("https://github.com/PX4/PX4-Autopilot/blob/release/1.14/"
                     "src/modules/commander/failure_detector/failure_detector_params.c")


@dataclass(frozen=True)
class Norm:
    feature: str
    autopilot: str          # "px4" | "ardupilot"
    kind: str
    direction: str
    healthy_edge: Optional[float]
    bad_edge: Optional[float]
    unit: str
    source: str
    url: str
    checked: str
    confidence: str         # spec §5.2: "low" | "medium" | "high"
    note: str = ""
    uses_detail: Optional[str] = None  # score a Feature.detail field instead of its value
    context: Optional[Dict[str, object]] = None  # CONTEXT: the evidence behind the context line


NORMS: Tuple[Norm, ...] = (
    # CONTEXT, not GRADED (amended 2026-10-27). On a survey of 23 real
    # vehicles from Flight Review's public index, measured with this
    # pipeline, 15 sat above 4.905 and 7 at or above 9.81: scored against
    # these lines, most first flights would be judged wrong. They are Flight
    # Review's plot shading; PX4's docs give no number. Reported as context,
    # scored only against the drone's own baseline (flight 3 on).
    # assets/docs/vibration-guide-survey.md (app repo).
    Norm("vib.metric_p50", "px4", CONTEXT, HIGHER_IS_WORSE, 4.905, 9.81, "m/s^2",
         "PX4 Flight Review 'Vibration Metrics' plot of vehicle_imu_status.accel_vibration_metric, "
         "background bands green/orange/red at 4.905 and 9.81",
         PX4_FLIGHT_REVIEW_PLOTS, CHECKED, "low",
         "a visual convention in Flight Review's code, not a documented PX4 limit",
         context={"vehicles": 23, "above_healthy": 15, "above_bad": 7, "median": 5.49,
                  "surveyed": "2026-10-27", "source": "Flight Review public index, multirotors since 2023",
                  "bias": "small; uploads skew to logs people wanted checked; metric depends on IMU and filter"}),
    # ArduPilot's lines are its own documented guidance, but no survey has
    # tested them; treated the same way so no absolute vibration level is ever
    # a verdict, and the context line quotes the guide's own words.
    Norm("vib.metric_p50", "ardupilot", CONTEXT, HIGHER_IS_WORSE, 30.0, 60.0, "m/s^2",
         "ArduPilot 'Measuring Vibration': below 30 normally acceptable, above 60 nearly always problems",
         AP_VIBRATION, CHECKED, "medium",
         "ArduPilot's VIBE is a different computation from PX4's metric; never compared across autopilots"),
    # vib.clip_rate and vib.imbalance were absolute norms (ArduPilot's
    # borrowed "<100 likely ok"; PX4's FD_IMB_PROP_THR 30). They are now
    # definitional facts (scoring.definitional): any clipping, or the PX4
    # detector's own trigger, is reported from flight one without making
    # Vibration a scored subsystem (spec §3.2, §3.4).
    Norm("gps.eph_p50", "px4", GATE, HIGHER_IS_WORSE, 3.0, None, "m",
         "PX4 v1.14 EKF2_REQ_EPH default 3.0 m ('Required EPH to use GPS')",
         PX4_EKF2_PARAMS_114, CHECKED, "medium",
         "the EKF's condition for starting to use GPS, read here as an in-flight health norm"),
    Norm("gps.nsats_p10", "px4", GATE, LOWER_IS_WORSE, 6.0, None, "count",
         "PX4 v1.14 EKF2_REQ_NSATS default 6 ('Required satellite count to use GPS')",
         PX4_EKF2_PARAMS_114, CHECKED, "medium",
         "the EKF's condition for starting to use GPS, read here as an in-flight health norm"),
    Norm("gps.sacc_p50", "px4", GATE, HIGHER_IS_WORSE, 0.5, None, "m/s",
         "PX4 v1.14 EKF2_REQ_SACC default 0.5 m/s ('Required speed accuracy to use GPS')",
         PX4_EKF2_PARAMS_114, CHECKED, "medium",
         "the EKF's condition for starting to use GPS, read here as an in-flight health norm"),
    Norm("pwr.v_cell_min", "px4", USAGE, LOWER_IS_WORSE, 3.6, None, "V/cell",
         "PX4 v1.14 BAT_V_EMPTY default 3.6 V ('Empty cell voltage (5C load)')",
         PX4_BATTERY_PARAMS_114, CHECKED, "high",
         "marks empty, not unhealthy; reported as 'flown to empty', never scored (spec §3.4)"),
)

_BY_KEY: Dict[Tuple[str, str], Norm] = {(n.feature, n.autopilot): n for n in NORMS}


def autopilot_of(source_format: str) -> str:
    """ULog is PX4; DataFlash and tlog are ArduPilot."""
    return "px4" if source_format == "ulog" else "ardupilot"


def norm_for(feature: str, autopilot: str) -> Optional[Norm]:
    return _BY_KEY.get((feature, autopilot))


@dataclass
class AbsoluteResult:
    score: Optional[float]   # None: no absolute score (no norm, usage-only, or below a bad-only trigger)
    status: str              # fine | watch | off | bad | no_norm | usage_ok | usage_flown_to_empty | not_computed
    norm: Optional[Norm]


def band_status(score: float) -> str:
    """The app's 85/70/50 bands (spec §4)."""
    if score >= 85:
        return "fine"
    if score >= 70:
        return "watch"
    if score >= 50:
        return "off"
    return "bad"


def graded_score(value: float, healthy: float, bad: float) -> float:
    """Higher-is-worse piecewise-linear score (spec §4.1): 0 -> 100,
    healthy -> 85, bad -> 50, 2 x bad -> 0. A healthy edge of 0 means only
    exactly zero is fully healthy."""
    if value <= healthy:
        return 100.0 if healthy == 0 else 100.0 - 15.0 * (value / healthy)
    if value <= bad:
        return 85.0 - 35.0 * (value - healthy) / (bad - healthy)
    return max(0.0, 50.0 - 50.0 * (value - bad) / bad)


def absolute_score(feature_key: str, value: Optional[float], autopilot: str,
                   detail: Optional[dict] = None) -> AbsoluteResult:
    norm = norm_for(feature_key, autopilot)
    if norm is None:
        return AbsoluteResult(None, "no_norm", None)
    x = value
    if norm.uses_detail:
        x = (detail or {}).get(norm.uses_detail)
    if x is None:
        return AbsoluteResult(None, "not_computed", norm)
    x = float(x)
    worse = (lambda edge: x > edge) if norm.direction == HIGHER_IS_WORSE else (lambda edge: x < edge)

    if norm.kind == CONTEXT:
        return AbsoluteResult(None, "context", norm)
    if norm.kind == USAGE:
        return AbsoluteResult(None, "usage_flown_to_empty" if worse(norm.healthy_edge) else "usage_ok", norm)
    if norm.kind == GATE:
        score = GATE_FAIL_SCORE if worse(norm.healthy_edge) else 100.0
        return AbsoluteResult(score, band_status(score), norm)
    if norm.kind == BAD_ONLY:
        if worse(norm.bad_edge) or x == norm.bad_edge:
            return AbsoluteResult(BAD_ONLY_SCORE, band_status(BAD_ONLY_SCORE), norm)
        return AbsoluteResult(None, "no_norm", norm)
    score = graded_score(x, norm.healthy_edge, norm.bad_edge)
    return AbsoluteResult(score, band_status(score), norm)


def is_absolute_bad(feature_key: str, value: Optional[float], autopilot: str, detail: Optional[dict] = None) -> bool:
    """Spec §2.3 admission gate (c): absolutely-bad values never join a baseline."""
    res = absolute_score(feature_key, value, autopilot, detail)
    return res.score is not None and res.score < 50.0
