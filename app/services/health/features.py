"""
Per-flight feature extraction (health spec 1.2, build step 3).

Every feature returns a Feature: either a value (or, for the one categorical
feature, a label) or null with a machine-readable reason. Never a silent
zero, never a guess. Reasons say what was missing so the report can tell the
user why, e.g. "requires_fifo_accel" or "insufficient_current_range".

Tiers follow spec 1.2: R1 reliable at default logging rates, R2 usable with
caveats, R3 needs a non-default logging profile or hardware and is never
required for a score.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.services.reality_gap.metrics import effective_sample_size
from app.services.reality_gap.parsers import ParsedLog, Signal
from app.services.reality_gap.spectral import welch_psd

from .windowing import FlightWindow


@dataclass
class Feature:
    key: str
    subsystem: str
    tier: str
    unit: str
    value: Optional[float] = None
    label: Optional[str] = None   # categorical result (act.pattern only)
    reason: Optional[str] = None  # set exactly when there is no value/label
    detail: Dict[str, object] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.reason is None


VIB, MOT, PWR, ATT, POS = "IMU & vibration", "Actuators", "Power", "Attitude & rates", "Position & velocity"

# Gates and constants, each from the spec.
MIN_SAMPLES = 10                 # fewer samples than this is not a distribution
CURRENT_RANGE_FRACTION = 0.30    # spec 1.2: p95-p5 of current >= 30% of its mean
SAT_THRESHOLD = 0.98             # spec 1.2: a motor at >= 98% normalised is saturated
PRE_TAKEOFF_S = 5.0              # "just before takeoff" window for pwr.sag_hover
HOVER_SAMPLE_S = 30.0            # first 30 s of steady flight for pwr.sag_hover
MANOEUVRE_RATE_RADPS = 0.3       # same 0.3 rad/s as the steady threshold: above it is commanded motion
MIN_MANOEUVRE_S = 10.0           # need >= 10 s of manoeuvring to measure tracking
BANDS_HZ = {"vib.band_lo": (0.0, 5.0), "vib.band_struct": (5.0, 20.0), "vib.band_hi": (20.0, None)}

# ponytail: pattern-classification cut-offs are judgement, not calibrated
# against real multi-drone data. Revisit with the spec 2.1 sigma floors once
# real histories exist.
PATTERN_BALANCED_SPREAD = 0.02   # spec 2.1 sigma_floor for act.spread
PATTERN_SINGLE_SHARE = 0.5       # one motor carries >= half the squared deviation
PATTERN_MODEL_R2 = 0.6           # a geometric explanation must explain >= 60%


def _null(key, subsystem, tier, unit, reason, **detail) -> Feature:
    return Feature(key, subsystem, tier, unit, reason=reason, detail=detail)


def _window_values(sig: Signal, t0: int, t1: int) -> np.ndarray:
    m = (sig.t_us >= t0) & (sig.t_us <= t1)
    return sig.values[m].astype(np.float64)


def _segment_mask(t_us: np.ndarray, segments: Sequence[Tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(t_us.size, dtype=bool)
    for a, b in segments:
        mask |= (t_us >= a) & (t_us <= b)
    return mask


def _segment_values(sig: Signal, segments) -> np.ndarray:
    return sig.values[_segment_mask(sig.t_us, segments)].astype(np.float64)


def _cells(parsed: ParsedLog) -> Tuple[Optional[int], str]:
    n = parsed.params.get("BAT1_N_CELLS")
    if n and n > 0:
        return int(n), "BAT1_N_CELLS"
    sig = parsed.signals.get("batt.cell_count")
    if sig is not None and sig.values.size and np.median(sig.values) > 0:
        return int(round(float(np.median(sig.values)))), "battery_status.cell_count"
    return None, "unknown"


# ---------------------------------------------------------------- vibration

def _vibration_metric(parsed: ParsedLog, w: FlightWindow) -> List[Feature]:
    keys = [("vib.metric_p50", 50), ("vib.metric_p95", 95)]
    x = parsed.signals.get("vibe.x")
    if x is None:
        return [_null(k, VIB, "R1", "m/s^2", "signal_absent:vibe") for k, _ in keys]
    # PX4 logs one combined metric (vibe.x only); ArduPilot logs per axis,
    # combined here as a vector magnitude per spec 1.2.
    y, z = parsed.signals.get("vibe.y"), parsed.signals.get("vibe.z")
    vals = _window_values(x, w.airborne_start_us, w.airborne_end_us)
    source = "px4_accel_vibration_metric"
    if y is not None and z is not None and y.values.size == x.values.size == z.values.size:
        vy = _window_values(y, w.airborne_start_us, w.airborne_end_us)
        vz = _window_values(z, w.airborne_start_us, w.airborne_end_us)
        vals = np.sqrt(vals ** 2 + vy ** 2 + vz ** 2)
        source = "ardupilot_vibe_magnitude"
    if vals.size < MIN_SAMPLES:
        return [_null(k, VIB, "R1", "m/s^2", "insufficient_samples", n=int(vals.size)) for k, _ in keys]
    n_eff = effective_sample_size(vals)
    return [Feature(k, VIB, "R1", "m/s^2", value=float(np.percentile(vals, q)),
                    detail={"n": int(vals.size), "n_eff": n_eff, "source": source}) for k, q in keys]


def _clip_rate(parsed: ParsedLog, w: FlightWindow) -> Feature:
    key = "vib.clip_rate"
    sigs = [parsed.signals.get(f"imu.clip{i}") for i in range(3)]
    if any(s is None for s in sigs):
        return _null(key, VIB, "R1", "1/min", "signal_absent:accel_clipping")
    total = 0.0
    for s in sigs:
        # Counters are cumulative: the clipping during the flight is the
        # difference between the last reading at or before landing and the
        # last reading at or before takeoff.
        before = s.values[s.t_us <= w.airborne_start_us]
        upto = s.values[s.t_us <= w.airborne_end_us]
        if upto.size == 0:
            return _null(key, VIB, "R1", "1/min", "insufficient_samples")
        start = float(before[-1]) if before.size else float(s.values[0])
        delta = float(upto[-1]) - start
        if delta < 0:
            return _null(key, VIB, "R1", "1/min", "counter_reset")
        total += delta
    return Feature(key, VIB, "R1", "1/min", value=total / (w.airborne_s / 60.0),
                   detail={"clip_events": int(total), "airborne_min": round(w.airborne_s / 60.0, 2)})


def _vibration_bands(parsed: ParsedLog, w: FlightWindow) -> List[Feature]:
    tiers = {"vib.band_lo": "R2", "vib.band_struct": "R2", "vib.band_hi": "R3"}
    unit = "dB re 1 (m/s^2)^2"
    if not w.steady_qualified:
        return [_null(k, VIB, tiers[k], unit, w.steady_reason or "insufficient_steady_flight") for k in BANDS_HZ]

    if all(f"accel_fifo.{a}" in parsed.signals for a in "xyz"):
        prefix, source, cutoff = "accel_fifo", "fifo", None  # pre-filter, no cutoff wall
    elif all(f"accel.{a}" in parsed.signals for a in "xyz"):
        prefix, source = "accel", "filtered"
        c = parsed.meta.imu_accel_cutoff
        cutoff = None if c is None or c <= 0 else float(c)  # 0 means the filter is disabled
    else:
        return [_null(k, VIB, tiers[k], unit, "signal_absent:accel") for k in BANDS_HZ]

    sx, sy, sz = (parsed.signals[f"{prefix}.{a}"] for a in "xyz")
    fs = sx.native_rate_hz
    if not fs or sx.values.size != sy.values.size or sx.values.size != sz.values.size:
        return [_null(k, VIB, tiers[k], unit, "accel_axes_misaligned") for k in BANDS_HZ]
    # Segment length for Welch: ~2.5 s, a power of two, so 0-5 Hz gets real bins.
    nperseg = 1 << int(np.floor(np.log2(max(16.0, fs * 2.5))))
    nperseg = min(nperseg, 4096)
    psd_sum, weight, freqs = None, 0, None
    for a, b in w.steady_segments:
        m = (sx.t_us >= a) & (sx.t_us <= b)
        if int(m.sum()) < nperseg:
            continue
        norm = np.sqrt(sx.values[m].astype(np.float64) ** 2 + sy.values[m] ** 2 + sz.values[m] ** 2)
        f, p = welch_psd(norm, fs, nperseg=nperseg)
        segs = max(1, (int(m.sum()) - nperseg) // (nperseg // 2) + 1)
        psd_sum = p * segs if psd_sum is None else psd_sum + p * segs
        weight += segs
        freqs = f
    if psd_sum is None:
        return [_null(k, VIB, tiers[k], unit, "steady_segments_shorter_than_welch_window",
                      nperseg=nperseg) for k in BANDS_HZ]
    psd = psd_sum / weight
    nyquist = fs / 2.0
    out = []
    for key, (lo, hi) in BANDS_HZ.items():
        hi_eff = nyquist if hi is None else hi
        detail = {"source": source, "fs_hz": round(fs, 1), "nperseg": nperseg, "band_hz": [lo, round(hi_eff, 1)]}
        if key == "vib.band_hi" and source == "filtered":
            out.append(_null(key, VIB, tiers[key], unit, "requires_fifo_accel", **detail))
            continue
        if lo >= nyquist:
            out.append(_null(key, VIB, tiers[key], unit, "band_above_nyquist", **detail))
            continue
        if cutoff is not None and lo >= cutoff:
            out.append(_null(key, VIB, tiers[key], unit, "band_above_filter_cutoff", **detail))
            continue
        if cutoff is not None and hi_eff > cutoff:
            detail["flag"] = "band_partially_filtered"
        mask = (freqs > lo) & (freqs < hi_eff)  # f > lo drops the DC bin
        if int(mask.sum()) < 2:
            out.append(_null(key, VIB, tiers[key], unit, "too_few_frequency_bins", **detail))
            continue
        power = float(np.trapezoid(psd[mask], freqs[mask]))
        if power <= 0:
            out.append(_null(key, VIB, tiers[key], unit, "no_power_in_band", **detail))
            continue
        out.append(Feature(key, VIB, tiers[key], unit, value=10.0 * np.log10(power), detail=detail))
    return out


def _imbalance(parsed: ParsedLog, w: FlightWindow) -> Feature:
    key = "vib.imbalance"
    sig = parsed.signals.get("fd.imbalance")
    if sig is None:
        return _null(key, VIB, "R2", "", "signal_absent:failure_detector_status")
    if not w.steady_qualified:
        return _null(key, VIB, "R2", "", w.steady_reason or "insufficient_steady_flight")
    vals = _segment_values(sig, w.steady_segments)
    if vals.size < MIN_SAMPLES:
        return _null(key, VIB, "R2", "", "insufficient_samples", n=int(vals.size))
    return Feature(key, VIB, "R2", "", value=float(np.median(vals)),
                   detail={"n": int(vals.size), "note": "PX4 metric; scale undocumented beyond FD_IMB_PROP_THR"})


# ---------------------------------------------------------------- motors

def _motor_series(parsed: ParsedLog) -> Tuple[List[Signal], str]:
    n = int(parsed.motor_info.get("motor_count") or 0)
    motors = [parsed.signals.get(f"act.motor{i}") for i in range(n)]
    if n and all(s is not None for s in motors):
        return motors, "actuator_motors"
    outs = [parsed.signals.get(f"act.out{i}") for i in range(n)]
    if n and all(s is not None for s in outs):
        return outs, "actuator_outputs"
    return [], "none"


def _classify_pattern(means: np.ndarray, params: Dict[str, float]) -> Feature:
    key = "act.pattern"
    n = means.size
    geo = []
    for i in range(n):
        px, py, km = (params.get(f"CA_ROTOR{i}_{a}") for a in ("PX", "PY", "KM"))
        if px is None or py is None or km is None:
            return _null(key, MOT, "R1", "", "geometry_unavailable")
        geo.append((px, py, km))
    geo = np.asarray(geo, dtype=np.float64)
    d = means - means.mean()
    total = float(np.dot(d, d))
    spread = float((means.max() - means.min()) / means.mean())
    detail = {"deviation": [round(float(v), 4) for v in d], "spread": round(spread, 4)}
    if spread < PATTERN_BALANCED_SPREAD or total <= 0:
        return Feature(key, MOT, "R1", "", label="balanced", detail=detail)

    order = np.argsort(d)[::-1]
    top, second = d[order[0]], d[order[1]]
    share = top ** 2 / total
    detail["top_share"] = round(float(share), 3)

    def r2(design: np.ndarray) -> float:
        coef, *_ = np.linalg.lstsq(design, d, rcond=None)
        res = d - design @ coef
        return 1.0 - float(np.dot(res, res)) / total

    r2_side = r2(geo[:, :2])                         # CG offset: deviation linear in position
    r2_spin = r2(np.sign(geo[:, 2]).reshape(-1, 1))  # yaw imbalance: deviation by spin direction
    detail.update({"r2_one_side": round(r2_side, 3), "r2_same_spin": round(r2_spin, 3)})
    high = [int(i) for i in np.flatnonzero(d > 0)]
    detail["high_motor_numbers"] = [i + 1 for i in high]

    if share >= PATTERN_SINGLE_SHARE and top > 0 and top >= 2 * max(second, 0.0):
        detail["motor_number"] = int(order[0]) + 1
        return Feature(key, MOT, "R1", "", label="single_motor_high", detail=detail)
    if max(r2_side, r2_spin) >= PATTERN_MODEL_R2:
        if r2_side >= r2_spin:
            coef, *_ = np.linalg.lstsq(geo[:, :2], d, rcond=None)
            # PX is forward, PY is right (FRD body frame); the high side is the
            # direction the deviation grows towards.
            detail["toward"] = "front" if abs(coef[0]) >= abs(coef[1]) and coef[0] > 0 else (
                "rear" if abs(coef[0]) >= abs(coef[1]) else ("right" if coef[1] > 0 else "left"))
            return Feature(key, MOT, "R1", "", label="one_side_high", detail=detail)
        detail["km_sign_of_high_set"] = "+" if np.mean(np.sign(geo[high, 2])) > 0 else "-"
        return Feature(key, MOT, "R1", "", label="same_spin_set_high", detail=detail)
    return Feature(key, MOT, "R1", "", label="mixed", detail=detail)


def _motor_features(parsed: ParsedLog, w: FlightWindow) -> List[Feature]:
    series, source = _motor_series(parsed)
    steady_keys = ["act.spread", "act.pattern"]
    if not series:
        reason = "no_motor_outputs"
        return ([_null("act.mean", MOT, "R1", "0-1", reason)] + [_null(k, MOT, "R1", "", reason) for k in steady_keys]
                + [_null("act.sat_frac", MOT, "R1", "fraction", reason)])
    out: List[Feature] = []

    # Saturation over the whole airborne window.
    t0, t1 = w.airborne_start_us, w.airborne_end_us
    stacked = [_window_values(s, t0, t1) for s in series]
    n = min(v.size for v in stacked)
    if n < MIN_SAMPLES:
        out.append(_null("act.sat_frac", MOT, "R1", "fraction", "insufficient_samples"))
    else:
        mat = np.vstack([v[:n] for v in stacked])
        out.append(Feature("act.sat_frac", MOT, "R1", "fraction",
                           value=float(np.mean(np.any(mat >= SAT_THRESHOLD, axis=0))),
                           detail={"n": n, "threshold": SAT_THRESHOLD, "source": source}))

    if not w.steady_qualified:
        reason = w.steady_reason or "insufficient_steady_flight"
        return ([_null("act.mean", MOT, "R1", "0-1", reason)] + [_null(k, MOT, "R1", "", reason) for k in steady_keys]
                + out)

    means = np.array([float(np.mean(_segment_values(s, w.steady_segments))) for s in series])
    out.append(Feature("act.mean", MOT, "R1", "0-1", value=float(means.mean()),
                       detail={"per_motor": [round(float(m), 4) for m in means], "source": source,
                               "motor_numbering": "1-based in detail, index i = motor i+1"}))
    if means.size < 2 or means.mean() <= 0:
        out.append(_null("act.spread", MOT, "R1", "", "too_few_motors"))
    else:
        out.append(Feature("act.spread", MOT, "R1", "", value=float((means.max() - means.min()) / means.mean()),
                           detail={"highest_motor_number": int(np.argmax(means)) + 1}))
    out.append(_classify_pattern(means, parsed.params))
    return out


def _hover(parsed: ParsedLog, w: FlightWindow) -> Feature:
    key = "act.hover"
    sig = parsed.signals.get("thrust.sp")
    if sig is None:
        return _null(key, MOT, "R1", "0-1", "signal_absent:thrust.sp")
    if not w.steady_qualified:
        return _null(key, MOT, "R1", "0-1", w.steady_reason or "insufficient_steady_flight")
    vals = _segment_values(sig, w.steady_segments)
    if vals.size < MIN_SAMPLES:
        return _null(key, MOT, "R1", "0-1", "insufficient_samples")
    return Feature(key, MOT, "R1", "0-1", value=float(np.median(vals)), detail={"n": int(vals.size)})


# ---------------------------------------------------------------- power

def _power_features(parsed: ParsedLog, w: FlightWindow) -> List[Feature]:
    cells, cell_src = _cells(parsed)
    v, i, mah = (parsed.signals.get(k) for k in ("batt.voltage_raw", "batt.current_raw", "batt.discharged_mah"))
    out: List[Feature] = []
    if v is None:
        reason = "signal_absent:battery_status.voltage_v"
        return [_null("pwr.r_cell", PWR, "R2", "mOhm", reason), _null("pwr.v_cell_min", PWR, "R1", "V", reason),
                _null("pwr.sag_hover", PWR, "R2", "V", reason), _cell_delta(parsed)]
    if cells is None:
        reason = "cell_count_unknown"
        return [_null("pwr.r_cell", PWR, "R2", "mOhm", reason), _null("pwr.v_cell_min", PWR, "R1", "V", reason),
                _null("pwr.sag_hover", PWR, "R2", "V", reason), _cell_delta(parsed)]

    t0, t1 = w.airborne_start_us, w.airborne_end_us
    out.append(_resistance(v, i, mah, t0, t1, cells, cell_src, parsed.params))

    vw = _window_values(v, t0, t1)
    if vw.size < MIN_SAMPLES:
        out.append(_null("pwr.v_cell_min", PWR, "R1", "V", "insufficient_samples"))
    else:
        out.append(Feature("pwr.v_cell_min", PWR, "R1", "V", value=float(vw.min()) / cells,
                           detail={"cells": cells, "cells_source": cell_src}))

    out.append(_sag(v, w, cells))
    out.append(_cell_delta(parsed))
    return out


def _resistance(v, i, mah, t0, t1, cells, cell_src, params) -> Feature:
    """
    Pack internal resistance from V = a + b*mAh - R*I over the airborne
    window (spec 1.2). The mAh term absorbs the open-circuit voltage falling
    as the pack drains, so R is read off current changes at roughly constant
    state of charge rather than confused with discharge.

    APPROXIMATION: open-circuit voltage is treated as linear in mAh over one
    flight. The real OCV curve bends near full and near empty, so a flight
    that covers both ends biases R. Raw voltage_v/current_a are used, not
    the _filtered fields, whose lag would bias the slope.
    """
    key = "pwr.r_cell"
    if i is None or mah is None:
        return _null(key, PWR, "R2", "mOhm", "signal_absent:battery_status.current_a/discharged_mah")
    if not (v.t_us.size == i.t_us.size == mah.t_us.size) or not np.array_equal(v.t_us, i.t_us) \
            or not np.array_equal(v.t_us, mah.t_us):
        return _null(key, PWR, "R2", "mOhm", "battery_fields_misaligned")
    m = (v.t_us >= t0) & (v.t_us <= t1)
    vv, ii, cc = (s.values[m].astype(np.float64) for s in (v, i, mah))
    if vv.size < 20:
        return _null(key, PWR, "R2", "mOhm", "insufficient_samples", n=int(vv.size))
    mean_i = float(np.mean(ii))
    p5, p95 = (float(np.percentile(ii, q)) for q in (5, 95))
    stats = {"current_mean_a": round(mean_i, 2), "current_p5_a": round(p5, 2), "current_p95_a": round(p95, 2)}
    if mean_i <= 0 or (p95 - p5) < CURRENT_RANGE_FRACTION * mean_i:
        return _null(key, PWR, "R2", "mOhm", "insufficient_current_range", **stats)
    design = np.column_stack([np.ones_like(ii), cc, -ii])
    coef, *_ = np.linalg.lstsq(design, vv, rcond=None)
    r_pack = float(coef[2])
    res = vv - design @ coef
    dof = max(1, vv.size - 3)
    sigma2 = float(np.dot(res, res)) / dof
    cov = sigma2 * np.linalg.inv(design.T @ design)
    se = float(np.sqrt(max(cov[2, 2], 0.0)))
    r2 = 1.0 - float(np.dot(res, res)) / float(np.dot(vv - vv.mean(), vv - vv.mean()))
    detail = dict(stats, r_pack_ohm=round(r_pack, 5), se_ohm=round(se, 5), r2=round(r2, 3), n=int(vv.size),
                  cells=cells, cells_source=cell_src, ocv_slope_v_per_mah=float(coef[1]),
                  bat_r_internal_param_ohm_per_cell=params.get("BAT1_R_INTERNAL"),
                  note="SE assumes independent samples; battery samples are autocorrelated, so it is optimistic")
    if r_pack <= 0:
        return _null(key, PWR, "R2", "mOhm", "non_physical_fit", **detail)
    return Feature(key, PWR, "R2", "mOhm", value=r_pack / cells * 1000.0, detail=detail)


def _sag(v: Signal, w: FlightWindow, cells: int) -> Feature:
    key = "pwr.sag_hover"
    pre = v.values[(v.t_us >= w.takeoff_us - int(PRE_TAKEOFF_S * 1e6)) & (v.t_us < w.takeoff_us)]
    if pre.size < 3:
        return _null(key, PWR, "R2", "V", "no_pre_takeoff_data", n=int(pre.size))
    if not w.steady_qualified:
        return _null(key, PWR, "R2", "V", w.steady_reason or "insufficient_steady_flight")
    # The first 30 s of steady flight, accumulated across segments.
    picked, remaining = [], HOVER_SAMPLE_S * 1e6
    for a, b in w.steady_segments:
        if remaining <= 0:
            break
        end = min(b, a + remaining)
        picked.append((a, end))
        remaining -= end - a
    hover = _segment_values(v, picked)
    if hover.size < 3:
        return _null(key, PWR, "R2", "V", "insufficient_samples")
    return Feature(key, PWR, "R2", "V", value=(float(np.median(pre)) - float(np.median(hover))) / cells,
                   detail={"v_rest": round(float(np.median(pre)), 3), "v_hover": round(float(np.median(hover)), 3),
                           "cells": cells, "note": "depends on the throttle used"})


def _cell_delta(parsed: ParsedLog) -> Feature:
    key = "pwr.cell_delta"
    sig = parsed.signals.get("batt.cell_delta")
    if sig is None:
        return _null(key, PWR, "R3", "V", "signal_absent:max_cell_voltage_delta")
    # A battery monitor without per-cell sensing reports exactly 0 throughout.
    if not np.any(sig.values != 0):
        return _null(key, PWR, "R3", "V", "no_per_cell_sensing")
    return Feature(key, PWR, "R3", "V", value=float(np.percentile(sig.values, 95)))


# ---------------------------------------------------------------- tracking

def _rate_tracking(parsed: ParsedLog, w: FlightWindow) -> List[Feature]:
    """
    Rate tracking error, over MANOEUVRING samples only (|setpoint| above the
    steady threshold), as RMS(gyro - setpoint) / RMS(setpoint).

    Why not the whole flight, as spec 1.2 first said: on the reference hex the
    whole-flight ratio was ~0.95 on roll and pitch, with setpoint-gyro
    correlation ~0.4 -- and low-passing both at 3 Hz barely changed it, so it
    isn't logging-rate aliasing. In steady flight the rate setpoint is the
    controller's small correction to disturbances and the gyro mostly measures
    the disturbance; they are not a command and its tracked response, so the
    ratio reflects wind and flying style rather than tuning. Restricted to
    commanded motion, correlation rose to ~0.7. Downgraded to R2 accordingly,
    and null unless there is enough manoeuvring to measure.

    The best-fit response lag is reported as a diagnostic (a large, slow
    airframe legitimately lags its setpoint, which inflates the ratio).
    """
    out = []
    for axis, gyro_key in (("roll", "gyro.x"), ("pitch", "gyro.y"), ("yaw", "gyro.z")):
        key = f"trk.rate_rms.{axis}"
        sp, g = parsed.signals.get(f"rate_sp.{axis}"), parsed.signals.get(gyro_key)
        if sp is None or g is None:
            out.append(_null(key, ATT, "R2", "ratio", "signal_absent:rate_setpoint_or_gyro"))
            continue
        m = (sp.t_us >= w.airborne_start_us) & (sp.t_us <= w.airborne_end_us)
        t = sp.t_us[m].astype(np.float64)
        rate = sp.native_rate_hz or 0.0
        if t.size < MIN_SAMPLES or rate <= 0:
            out.append(_null(key, ATT, "R2", "ratio", "insufficient_samples"))
            continue
        spv = sp.values[m].astype(np.float64)
        gv = np.interp(t, g.t_us.astype(np.float64), g.values.astype(np.float64))
        active = np.abs(spv) > MANOEUVRE_RATE_RADPS
        need = int(MIN_MANOEUVRE_S * rate)
        detail = {"active_samples": int(active.sum()), "active_fraction": round(float(active.mean()), 3),
                  "rate_hz": round(rate, 1), "manoeuvre_threshold_rad_s": MANOEUVRE_RATE_RADPS}
        if int(active.sum()) < need:
            out.append(_null(key, ATT, "R2", "ratio", "insufficient_manoeuvring", needed_samples=need, **detail))
            continue
        err = gv[active] - spv[active]
        rms_sp = float(np.sqrt(np.mean(spv[active] ** 2)))
        rel = float(np.sqrt(np.mean(err ** 2))) / rms_sp
        # Diagnostic: the lag (0-300 ms) at which gyro best matches the setpoint.
        lags = range(0, max(1, int(0.3 * rate)))
        errs = [float(np.sqrt(np.mean((np.roll(gv, -k)[active] - spv[active]) ** 2))) for k in lags]
        k = int(np.argmin(errs))
        detail.update({"rms_setpoint_deg_s": round(float(np.degrees(rms_sp)), 3),
                       "best_lag_ms": round(k / rate * 1000.0, 1),
                       "ratio_at_best_lag": round(errs[k] / rms_sp, 3)})
        out.append(Feature(key, ATT, "R2", "ratio", value=rel, detail=detail))
    return out


def _attitude_error(parsed: ParsedLog, w: FlightWindow) -> Feature:
    key = "trk.att_err_p95"
    q = [parsed.signals.get(f"att.{c}") for c in ("qw", "qx", "qy", "qz")]
    qd = [parsed.signals.get(f"att_sp.{c}") for c in ("qw", "qx", "qy", "qz")]
    if any(s is None for s in q + qd):
        return _null(key, ATT, "R1", "deg", "signal_absent:vehicle_attitude_setpoint")
    t_sp = qd[0].t_us
    m = (t_sp >= w.airborne_start_us) & (t_sp <= w.airborne_end_us)
    if int(m.sum()) < MIN_SAMPLES:
        return _null(key, ATT, "R1", "deg", "insufficient_samples")
    tt = t_sp[m].astype(np.float64)
    sp = np.vstack([s.values[m].astype(np.float64) for s in qd])
    act = np.vstack([np.interp(tt, s.t_us.astype(np.float64), s.values.astype(np.float64)) for s in q])
    sp /= np.linalg.norm(sp, axis=0)
    act /= np.linalg.norm(act, axis=0)
    # Rotation angle between the two attitudes: 2*acos(|<q_sp, q>|); the
    # absolute value handles q and -q describing the same attitude.
    dot = np.clip(np.abs(np.sum(sp * act, axis=0)), 0.0, 1.0)
    ang = np.degrees(2.0 * np.arccos(dot))
    return Feature(key, ATT, "R1", "deg", value=float(np.percentile(ang, 95)),
                   detail={"n": int(ang.size), "median_deg": round(float(np.median(ang)), 3)})


# ---------------------------------------------------------------- position

def _gps_estimator(parsed: ParsedLog, w: FlightWindow) -> List[Feature]:
    specs = [
        ("gps.eph_p50", "gps.eph", 50, "R1", "m"),
        ("gps.nsats_p10", "gps.nsats", 10, "R1", "count"),
        ("gps.sacc_p50", "gps.sacc", 50, "R1", "m/s"),
        ("gps.noise", "gps.noise", 50, "R2", "receiver units"),
        ("gps.jamming", "gps.jamming", 50, "R2", "receiver units"),
    ]
    out = []
    for key, sig_key, q, tier, unit in specs:
        sig = parsed.signals.get(sig_key)
        if sig is None:
            out.append(_null(key, POS, tier, unit, f"signal_absent:{sig_key}"))
            continue
        vals = _window_values(sig, w.airborne_start_us, w.airborne_end_us)
        if vals.size < MIN_SAMPLES:
            out.append(_null(key, POS, tier, unit, "insufficient_samples", n=int(vals.size)))
            continue
        out.append(Feature(key, POS, tier, unit, value=float(np.percentile(vals, q)),
                           detail={"n": int(vals.size), "n_eff": effective_sample_size(vals)}))

    ratios = {}
    for name in ("vel", "pos", "hgt", "mag"):
        sig = parsed.signals.get(f"est.{name}_tr")
        if sig is not None:
            vals = _window_values(sig, w.airborne_start_us, w.airborne_end_us)
            if vals.size >= MIN_SAMPLES:
                ratios[name] = float(np.percentile(vals, 95))
    if not ratios:
        out.append(_null("est.test_ratio_p95", POS, "R1", "ratio", "signal_absent:estimator_status"))
    else:
        worst = max(ratios, key=ratios.get)
        out.append(Feature("est.test_ratio_p95", POS, "R1", "ratio", value=ratios[worst],
                           detail={"per_ratio_p95": {k: round(v, 3) for k, v in ratios.items()}, "worst": worst}))
    return out


# ---------------------------------------------------------------- driver

def extract_features(parsed: ParsedLog, window: FlightWindow) -> List[Feature]:
    """Every spec 1.2 feature for one flight. An unqualified flight returns
    every feature as null with the windowing reasons, never partial guesses."""
    feats: List[Feature] = []
    if window.qualified:
        feats += _vibration_metric(parsed, window)
        feats.append(_clip_rate(parsed, window))
        feats += _vibration_bands(parsed, window)
        feats.append(_imbalance(parsed, window))
        feats += _motor_features(parsed, window)
        feats.append(_hover(parsed, window))
        feats += _power_features(parsed, window)
        feats += _rate_tracking(parsed, window)
        feats.append(_attitude_error(parsed, window))
        feats += _gps_estimator(parsed, window)
        return feats
    reason = "flight_not_qualified:" + ";".join(window.reasons)
    for key, sub, tier, unit in FEATURE_CATALOGUE:
        feats.append(_null(key, sub, tier, unit, reason))
    return feats


# Every feature key the extractor produces, with subsystem, tier and unit.
FEATURE_CATALOGUE = [
    ("vib.metric_p50", VIB, "R1", "m/s^2"), ("vib.metric_p95", VIB, "R1", "m/s^2"),
    ("vib.clip_rate", VIB, "R1", "1/min"),
    ("vib.band_lo", VIB, "R2", "dB re 1 (m/s^2)^2"), ("vib.band_struct", VIB, "R2", "dB re 1 (m/s^2)^2"),
    ("vib.band_hi", VIB, "R3", "dB re 1 (m/s^2)^2"), ("vib.imbalance", VIB, "R2", ""),
    ("act.sat_frac", MOT, "R1", "fraction"), ("act.mean", MOT, "R1", "0-1"), ("act.spread", MOT, "R1", ""),
    ("act.pattern", MOT, "R1", ""), ("act.hover", MOT, "R1", "0-1"),
    ("pwr.r_cell", PWR, "R2", "mOhm"), ("pwr.v_cell_min", PWR, "R1", "V"), ("pwr.sag_hover", PWR, "R2", "V"),
    ("pwr.cell_delta", PWR, "R3", "V"),
    ("trk.rate_rms.roll", ATT, "R2", "ratio"), ("trk.rate_rms.pitch", ATT, "R2", "ratio"),
    ("trk.rate_rms.yaw", ATT, "R2", "ratio"), ("trk.att_err_p95", ATT, "R1", "deg"),
    ("gps.eph_p50", POS, "R1", "m"), ("gps.nsats_p10", POS, "R1", "count"), ("gps.sacc_p50", POS, "R1", "m/s"),
    ("gps.noise", POS, "R2", "receiver units"), ("gps.jamming", POS, "R2", "receiver units"),
    ("est.test_ratio_p95", POS, "R1", "ratio"),
]
