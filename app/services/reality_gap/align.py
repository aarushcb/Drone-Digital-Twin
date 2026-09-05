"""
D1 spec 5.3 (time alignment), 5.4 (resampling) and 5.5 (segment selection).

THIS FILE IS THE MODULE'S MAIN HONESTY GUARD, NOT JUST PLUMBING.
Spec 5.3 says it outright: "NRMSE between two unrelated flights is a
large, confident, meaningless number." Everything here exists so that a
point-wise metric is only ever computed on two series that genuinely
correspond in time, and so that when they do NOT correspond the pipeline
says so loudly (alignment_quality="coarse_only") instead of returning a
confident number nobody should trust.

THREE SEPARATE PROBLEMS, SOLVED IN ORDER:
1. The two logs start at different points in their own boot clocks, and
   the mission begins at a different offset within each recording ->
   coarse anchor + fine cross-correlation (5.3).
2. The two logs sample at different, non-uniform rates -> resample onto
   one common grid (5.4), while carefully NOT pretending upsampled
   points are new information.
3. Ground time flatters the simulator (both logs sit still and agree
   beautifully), so metrics must be computed on the armed/airborne
   portion only (5.5).

DETERMINISM (spec 10.3 item 16 -- "the same log pair processed twice
produces byte-identical metrics; no RNG anywhere in the pipeline without
a fixed seed"): nothing in this file uses randomness at all. Every
operation is a deterministic array transform.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal

from .parsers import ParsedLog, Signal

# --- spec 5.4 [DECISION] canonical grid ---
CANONICAL_RATE_HZ = 50.0
# --- spec 5.4 [DECISION] high-rate spectral path ---
HIGH_RATE_CAP_HZ = 500.0
HIGH_RATE_SIGNALS = ("accel.x", "accel.y", "accel.z", "vibe.x", "vibe.y", "vibe.z")

# --- spec 5.3 [DECISION] fine alignment ---
FINE_SEARCH_S = 10.0
FINE_CORRELATION_GATE = 0.5
FINE_BANDPASS_HZ = (0.2, 5.0)

# --- spec 5.5 [DECISION] segment selection ---
AIRBORNE_ALT_M = 0.5
TAKEOFF_ACCEL_EXCESS = 1.0     # m/s^2 above 1 g
TAKEOFF_SUSTAIN_S = 0.5
PHASE_MIN_SAMPLES = 250        # 5 s at 50 Hz
PHASE_VZ_THRESHOLD = 0.5       # m/s, NED (negative = climbing)

# --- spec 9 case 1 overlap gates ---
MIN_OVERLAP_S = 20.0
SHORT_OVERLAP_FRACTION = 0.40

# Signals for which interpolation is nonsense -- spec 5.4: "interpolating
# a fix type produces fix type 2.7". These use zero-order hold instead.
ZERO_ORDER_HOLD_SIGNALS = ("gps.fix_type", "gps.nsats")

# Spec 9 case 5: below this native rate a signal is marked low_rate, gets
# no spectral metrics, and is excluded from the IMU/vibration subsystem.
LOW_RATE_HZ = 5.0


class AlignmentError(Exception):
    """Raised when the two logs cannot be compared at all (spec 9 cases 1, 9)."""


@dataclass
class AlignmentResult:
    """Everything the report needs to explain HOW the two logs were lined up.

    `quality` is either "fine" (cross-correlation succeeded and was
    applied) or "coarse_only" (gate failed; spec 5.3 distribution-only
    mode). The distinction drives whether point-wise metrics are
    trustworthy, so it is carried explicitly rather than inferred from
    the correlation number by each consumer.
    """

    quality: str
    coarse_anchor_real_us: int
    coarse_anchor_sim_us: int
    coarse_method: str
    fine_shift_s: float
    fine_correlation: Optional[float]
    overlap_s: float
    real_duration_s: float
    sim_duration_s: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class ResampledPair:
    """One signal, both sides, on the shared grid, plus the native samples.

    BOTH representations are kept on purpose (spec 5.4): point-wise
    metrics (NRMSE, bias, correlation) need the common grid, but
    distribution and spectral metrics must run on the NATIVE samples of
    each log, because upsampled points are not new information and
    feeding them to a KS test inflates N and destroys the p-value's
    meaning.
    """

    key: str
    t_grid_s: np.ndarray
    real_grid: np.ndarray
    sim_grid: np.ndarray
    real_native: np.ndarray
    sim_native: np.ndarray
    native_rate_real: Optional[float]
    native_rate_sim: Optional[float]
    flags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 5.3 -- coarse anchor
# ---------------------------------------------------------------------------


def find_arm_anchor(parsed: ParsedLog) -> Tuple[Optional[int], str]:
    """
    Spec 5.3 step 1. Returns (anchor_timestamp_us, method).

    Preference order, and why:
    1. An explicit arm event is the ground truth for "the mission started
       here" and is what the spec asks for first.
    2. Takeoff detection (|accel.z| deviating from 1 g by more than
       1.0 m/s^2, sustained 0.5 s) is the documented fallback for logs
       with no arm record.
    3. First sample, with a warning, if neither exists.

    NOTE ON THE ARM EVENT IN PASS 1: the canonical schema (spec 5.2) has
    no armed/disarmed signal -- vehicle_status.arming_state and
    ArduPilot's EV/MODE messages are not among the 25 canonical keys, so
    the parser does not currently surface them. Rather than silently
    always using the fallback and reporting a method the code never
    actually took, this checks for an optional 'status.armed' signal (so
    the branch is real and tested via a synthetic fixture) and otherwise
    falls through to takeoff detection. Adding arming_state to the
    canonical schema is tracked as a pass-2 item.
    """
    armed = parsed.signals.get("status.armed")
    if armed is not None and armed.values.size:
        transitions = np.nonzero(armed.values > 0.5)[0]
        if transitions.size:
            return int(armed.t_us[transitions[0]]), "arm_event"

    accel_z = parsed.signals.get("accel.z")
    if accel_z is not None and accel_z.values.size > 1:
        rate = accel_z.native_rate_hz or CANONICAL_RATE_HZ
        need = max(1, int(TAKEOFF_SUSTAIN_S * rate))
        excess = np.abs(np.abs(accel_z.values) - 9.81) > TAKEOFF_ACCEL_EXCESS
        if excess.size >= need:
            # A run of `need` consecutive True values -> the convolution of
            # the boolean series with a ones-kernel hits `need` exactly at
            # the END of the first qualifying run.
            run = np.convolve(excess.astype(np.int32), np.ones(need, dtype=np.int32), mode="valid")
            hits = np.nonzero(run >= need)[0]
            if hits.size:
                return int(accel_z.t_us[int(hits[0])]), "takeoff_detect"

    first = _earliest_timestamp(parsed)
    if first is None:
        return None, "none"
    return first, "first_sample"


def _earliest_timestamp(parsed: ParsedLog) -> Optional[int]:
    starts = [int(s.t_us[0]) for s in parsed.signals.values() if s.t_us.size]
    return min(starts) if starts else None


def _latest_timestamp(parsed: ParsedLog) -> Optional[int]:
    ends = [int(s.t_us[-1]) for s in parsed.signals.values() if s.t_us.size]
    return max(ends) if ends else None


# ---------------------------------------------------------------------------
# 5.3 -- fine alignment
# ---------------------------------------------------------------------------


def _bandpass(values: np.ndarray, rate_hz: float) -> np.ndarray:
    """
    0.2-5 Hz band-pass (spec 5.3 step 2) before cross-correlating.

    WHY BAND-PASS AT ALL: the DC/very-low-frequency content of an attitude
    trace is dominated by trim offset, which is a constant and correlates
    with everything; the high-frequency end is sensor noise, which
    correlates with nothing. The 0.2-5 Hz band is where actual manoeuvre
    structure lives, which is the only thing that can tell you these are
    the same mission.

    Falls back to mean-removal when the sample rate is too low for the
    requested band to exist (Nyquist below the 5 Hz upper edge) -- a
    filter designed with a corner above Nyquist is not just inaccurate,
    scipy raises on it.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size < 16 or rate_hz <= 0:
        return values - values.mean() if values.size else values
    nyquist = rate_hz / 2.0
    low, high = FINE_BANDPASS_HZ
    if high >= nyquist:
        high = nyquist * 0.9
    if low >= high:
        return values - values.mean()
    sos = sp_signal.butter(2, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    return sp_signal.sosfiltfilt(sos, values)


def _normalised_cross_correlation(
    a: np.ndarray, b: np.ndarray, max_lag: int
) -> Tuple[int, float]:
    """
    Returns (lag_in_samples, peak_normalised_correlation) maximising the
    normalised cross-correlation of `b` against `a`, searched over
    +/- max_lag.

    Normalisation is by the product of the two signals' L2 norms after
    mean removal, which bounds the result to [-1, 1] and makes the
    correlation directly comparable to the spec's 0.5 gate. A raw
    (unnormalised) correlate() peak would scale with signal amplitude and
    could not be thresholded meaningfully.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(a.size, b.size)
    if n < 4:
        return 0, 0.0
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom <= 0:
        return 0, 0.0
    full = sp_signal.correlate(a, b, mode="full", method="auto") / denom
    lags = sp_signal.correlation_lags(a.size, b.size, mode="full")
    window = np.abs(lags) <= max_lag
    if not np.any(window):
        return 0, 0.0
    windowed = full[window]
    windowed_lags = lags[window]
    best = int(np.argmax(windowed))
    return int(windowed_lags[best]), float(windowed[best])


def _pick_reference_signal(real: ParsedLog, sim: ParsedLog) -> Optional[str]:
    """
    Spec 5.3: use att.roll, or att.pitch if roll's variance is the lower
    of the two. Rationale for preferring the higher-variance axis: the
    axis that actually moved during the mission carries the manoeuvre
    structure the correlation needs; correlating a near-constant axis
    finds a spurious peak.
    """
    candidates = []
    for key in ("att.roll", "att.pitch"):
        if key in real.signals and key in sim.signals:
            variance = float(np.var(real.signals[key].values))
            candidates.append((variance, key))
    if not candidates:
        return None
    return max(candidates)[1]


# ---------------------------------------------------------------------------
# 5.4 -- resampling
# ---------------------------------------------------------------------------


def resample_to_grid(
    sig: Signal, t0_us: int, t1_us: int, rate_hz: float, zero_order_hold: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample one signal onto a uniform grid spanning [t0_us, t1_us].

    Zero-order hold (previous-value) is used for enum/count signals --
    spec 5.4 is explicit that interpolating a GPS fix type into 2.7 is
    nonsense. np.searchsorted gives the previous-sample index directly.

    Samples outside the signal's own measured span are NOT extrapolated:
    np.interp clamps to the endpoint values, which would invent a
    perfectly flat reading for a period the sensor never covered. Those
    positions are returned as NaN so downstream code can exclude them
    rather than average over fabricated data.
    """
    n = max(2, int(round((t1_us - t0_us) / 1e6 * rate_hz)) + 1)
    grid_us = np.linspace(t0_us, t1_us, n)
    if sig.t_us.size == 0:
        return grid_us, np.full(n, np.nan)

    t_src = sig.t_us.astype(np.float64)
    v_src = sig.values.astype(np.float64)

    if zero_order_hold:
        idx = np.searchsorted(t_src, grid_us, side="right") - 1
        out = np.where(idx >= 0, v_src[np.clip(idx, 0, v_src.size - 1)], np.nan)
    else:
        out = np.interp(grid_us, t_src, v_src)

    outside = (grid_us < t_src[0]) | (grid_us > t_src[-1])
    out = np.asarray(out, dtype=np.float64)
    out[outside] = np.nan
    return grid_us, out


# ---------------------------------------------------------------------------
# 5.5 -- segment selection
# ---------------------------------------------------------------------------


def airborne_window(
    parsed: ParsedLog, anchor_us: int
) -> Tuple[int, Optional[int], List[str]]:
    """
    Spec 5.5: from the arm anchor to first disarm, trimmed to samples
    where alt.rel > 0.5 m when altitude is available.

    Returns (start_us, end_us_or_None, warnings). end_us None means "run
    to the end of the log" -- there is no disarm signal in the canonical
    schema in this pass (same gap noted in find_arm_anchor).
    """
    warnings: List[str] = []
    alt = parsed.signals.get("alt.rel")
    if alt is None or alt.t_us.size == 0:
        return anchor_us, None, warnings

    mask = (alt.t_us >= anchor_us) & (alt.values > AIRBORNE_ALT_M)
    if not np.any(mask):
        # Spec 9 case 9: armed but never above 0.5 m -> proceed on the
        # armed window rather than failing, but say so.
        warnings.append("ground_only")
        return anchor_us, None, warnings

    idx = np.nonzero(mask)[0]
    return int(alt.t_us[idx[0]]), int(alt.t_us[idx[-1]]), warnings


def classify_phases(vz_grid: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Spec 5.5 [DECISION]: phase boundaries by vertical velocity in NED,
    where negative vz is climbing. Returns boolean masks; a phase with
    fewer than PHASE_MIN_SAMPLES samples is omitted entirely rather than
    reported from too little data.
    """
    with np.errstate(invalid="ignore"):
        masks = {
            "climb": vz_grid < -PHASE_VZ_THRESHOLD,
            "cruise": np.abs(vz_grid) <= PHASE_VZ_THRESHOLD,
            "descent": vz_grid > PHASE_VZ_THRESHOLD,
        }
    return {name: m for name, m in masks.items() if int(np.nansum(m)) >= PHASE_MIN_SAMPLES}


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def align_logs(real: ParsedLog, sim: ParsedLog) -> AlignmentResult:
    """
    Full spec 5.3 two-step alignment. Does not modify either ParsedLog --
    returns the offsets, which resample_pair() then applies.
    """
    warnings: List[str] = []

    real_anchor, real_method = find_arm_anchor(real)
    sim_anchor, sim_method = find_arm_anchor(sim)
    if real_anchor is None or sim_anchor is None:
        missing = real.meta.filename if real_anchor is None else sim.meta.filename
        raise AlignmentError(f"no flight detected in {missing}")
    if "first_sample" in (real_method, sim_method):
        warnings.append("coarse_alignment_fallback_first_sample")

    real_end = _latest_timestamp(real)
    sim_end = _latest_timestamp(sim)
    real_duration = (real_end - real_anchor) / 1e6 if real_end else 0.0
    sim_duration = (sim_end - sim_anchor) / 1e6 if sim_end else 0.0

    fine_shift_s = 0.0
    correlation: Optional[float] = None
    quality = "coarse_only"

    ref_key = _pick_reference_signal(real, sim)
    if ref_key is not None:
        overlap_guess = max(1.0, min(real_duration, sim_duration))
        grid_n = int(overlap_guess * CANONICAL_RATE_HZ)
        if grid_n >= 32:
            r_sig, s_sig = real.signals[ref_key], sim.signals[ref_key]
            _, r_grid = resample_to_grid(
                r_sig, real_anchor, real_anchor + int(overlap_guess * 1e6), CANONICAL_RATE_HZ, False
            )
            _, s_grid = resample_to_grid(
                s_sig, sim_anchor, sim_anchor + int(overlap_guess * 1e6), CANONICAL_RATE_HZ, False
            )
            valid = np.isfinite(r_grid) & np.isfinite(s_grid)
            if int(valid.sum()) >= 32:
                r_f = _bandpass(r_grid[valid], CANONICAL_RATE_HZ)
                s_f = _bandpass(s_grid[valid], CANONICAL_RATE_HZ)
                max_lag = int(FINE_SEARCH_S * CANONICAL_RATE_HZ)
                lag, correlation = _normalised_cross_correlation(r_f, s_f, max_lag)
                if correlation is not None and correlation >= FINE_CORRELATION_GATE:
                    fine_shift_s = lag / CANONICAL_RATE_HZ
                    quality = "fine"
                else:
                    # Spec 5.3 [DECISION] fine-alignment gate: do NOT apply
                    # a shift we don't believe. This is the guard that stops
                    # the module reporting a confident NRMSE between two
                    # flights that were never the same mission.
                    warnings.append("fine_alignment_below_gate")

    overlap_s = max(0.0, min(real_duration, sim_duration - fine_shift_s))
    if overlap_s < MIN_OVERLAP_S:
        raise AlignmentError(
            f"only {overlap_s:.0f} s of overlap between the two logs; need at least "
            f"{MIN_OVERLAP_S:.0f} s"
        )
    shorter = min(real_duration, sim_duration)
    if shorter > 0 and overlap_s < SHORT_OVERLAP_FRACTION * shorter:
        warnings.append("short_overlap")

    return AlignmentResult(
        quality=quality,
        coarse_anchor_real_us=real_anchor,
        coarse_anchor_sim_us=sim_anchor,
        coarse_method=f"real:{real_method},sim:{sim_method}",
        fine_shift_s=fine_shift_s,
        fine_correlation=correlation,
        overlap_s=overlap_s,
        real_duration_s=real_duration,
        sim_duration_s=sim_duration,
        warnings=warnings,
    )


def resample_pair(
    real: ParsedLog, sim: ParsedLog, alignment: AlignmentResult
) -> Dict[str, ResampledPair]:
    """
    Apply the alignment and put every shared signal on the canonical grid
    (spec 5.4), restricted to the airborne window (spec 5.5).

    Signals present on only one side are simply absent from the result --
    spec 9 cases 2 and 3 require them to be reported as excluded, which is
    the caller's job (it needs to distinguish "missing on one side" from
    "absent on both", and only the caller knows the full expected key
    set).
    """
    real_start, real_end, real_warn = airborne_window(real, alignment.coarse_anchor_real_us)
    sim_start, _, _ = airborne_window(sim, alignment.coarse_anchor_sim_us)
    alignment.warnings.extend(w for w in real_warn if w not in alignment.warnings)

    span_us = int(alignment.overlap_s * 1e6)
    if real_end is not None:
        span_us = min(span_us, int(real_end - real_start))
    if span_us <= 0:
        raise AlignmentError("no usable airborne overlap between the two logs")

    # The fine shift is applied to the SIM side only (spec 5.3: "apply that
    # shift to all sim signals"), keeping the real log as the fixed
    # reference frame the report is expressed in.
    sim_start_shifted = int(sim_start + alignment.fine_shift_s * 1e6)

    out: Dict[str, ResampledPair] = {}
    shared = sorted(set(real.signals) & set(sim.signals))
    for key in shared:
        r_sig, s_sig = real.signals[key], sim.signals[key]
        zoh = key in ZERO_ORDER_HOLD_SIGNALS

        t_us, r_grid = resample_to_grid(
            r_sig, real_start, real_start + span_us, CANONICAL_RATE_HZ, zoh
        )
        _, s_grid = resample_to_grid(
            s_sig, sim_start_shifted, sim_start_shifted + span_us, CANONICAL_RATE_HZ, zoh
        )

        flags: List[str] = []
        rate_r, rate_s = r_sig.native_rate_hz, s_sig.native_rate_hz
        if (rate_r or 0) < LOW_RATE_HZ or (rate_s or 0) < LOW_RATE_HZ:
            flags.append("low_rate")
        # Spec 5.4: "never upsample beyond a signal's own native rate" --
        # the grid values still exist for point-wise metrics, but the flag
        # tells the metrics layer that distribution/spectral work must use
        # the native arrays carried alongside.
        if (rate_r or 0) < CANONICAL_RATE_HZ or (rate_s or 0) < CANONICAL_RATE_HZ:
            flags.append("upsampled_to_grid")

        real_native = _native_window(r_sig, real_start, real_start + span_us)
        sim_native = _native_window(s_sig, sim_start_shifted, sim_start_shifted + span_us)

        out[key] = ResampledPair(
            key=key,
            t_grid_s=(t_us - t_us[0]) / 1e6,
            real_grid=r_grid.astype(np.float32),
            sim_grid=s_grid.astype(np.float32),
            real_native=real_native,
            sim_native=sim_native,
            native_rate_real=rate_r,
            native_rate_sim=rate_s,
            flags=flags,
        )
    return out


def _native_window(sig: Signal, t0_us: int, t1_us: int) -> np.ndarray:
    """The signal's own untouched samples inside the analysis window --
    what distribution and spectral metrics must use (spec 5.4)."""
    mask = (sig.t_us >= t0_us) & (sig.t_us <= t1_us)
    return sig.values[mask].astype(np.float32)
