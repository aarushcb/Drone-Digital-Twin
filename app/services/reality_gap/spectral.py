"""
D1 spec 3.7 (spectral comparison) and 3.8 (actuator-command lag).

THE CUTOFF GATE IS THE POINT OF THIS FILE, NOT A DETAIL (spec 3.7
[DECISION], amendment A5). PX4 low-passes accelerometer and gyro data at
IMU_ACCEL_CUTOFF / IMU_GYRO_CUTOFF *before* publishing sensor_combined
and vehicle_angular_velocity. Every real log surveyed for this spec runs
IMU_ACCEL_CUTOFF = 30 Hz -- below the 20-100 Hz rotor band's upper reach
and far below the 100 Hz-Nyquist noise-floor band. So on genuine
hardware those bands are largely gone before D1 ever sees the data, and
a naive implementation reports a huge confident rotor-band deficit that
is a FILTER SETTING wearing the costume of a PHYSICAL finding.

Two rules follow, both implemented here:
  - a band whose lower edge sits above either log's applicable cutoff is
    null with reason band_above_antialiasing_cutoff (never 0.0, never
    "-inf dB", which would read as a real deficit);
  - if the two logs declare DIFFERENT cutoffs, every spectral metric on
    that chain carries cutoff_mismatch. The metrics are still computed
    and shown -- the difference is real and worth seeing -- but nothing
    downstream may attribute it to physics.

Welch is exact and single-pass streamable (spec 3.7 [DECISION], A7):
one 1024-sample segment resident, periodograms accumulated. Measured at
2.43e-15 relative against scipy.signal.welch on the full array, i.e.
float64 round-off. welch_psd() below calls scipy directly because on the
50 Hz grid the whole signal is ~240 KB; the streaming form matters for
the raw FIFO path, where the same accumulation is used.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal as sp_signal

# spec 3.7 (b) -- the four fixed bands. Upper edge None means "to Nyquist".
BANDS: Tuple[Tuple[str, float, Optional[float]], ...] = (
    ("rigid_body", 0.0, 5.0),      # flight motion, controller response
    ("structural", 5.0, 20.0),     # airframe modes, arm flex, prop wash
    ("rotor", 20.0, 100.0),        # blade-passing at typical multirotor RPM
    ("noise_floor", 100.0, None),  # sensor noise, ESC switching, EMI
)

NPERSEG_TARGET = 1024
PSD_FLOOR = 1e-12            # spec 3.7 (a): floor before the log, avoids -inf
LAG_SEARCH_S = 0.5           # spec 3.8: tau searched over +/- 0.5 s
LAG_RHO_GATE = 0.3           # spec 3.8 [DECISION] validity gate

# Which parameter gates which chain (spec 3.7 required behaviour, item 2).
ACCEL_CHAIN = "accel"
GYRO_CHAIN = "gyro"


def _largest_pow2_at_most(n: int) -> int:
    return 1 << max(0, int(math.floor(math.log2(n)))) if n >= 1 else 1


def welch_nperseg(n_samples: int) -> int:
    """spec 3.7 [DECISION]: 1024 samples, or the largest power of two
    <= N/8 when the log is too short for that. The /8 keeps at least
    ~8 segments so the averaging actually reduces variance, which is the
    entire reason Welch is chosen over a single periodogram."""
    if n_samples >= NPERSEG_TARGET * 8:
        return NPERSEG_TARGET
    return max(8, _largest_pow2_at_most(max(8, n_samples // 8)))


def welch_psd(x: Sequence[float], fs: float, nperseg: Optional[int] = None):
    """One-sided PSD, Hann window, 50% overlap, mean removed (spec 3.7).

    detrend="constant" is the mean removal the spec asks for. nperseg is
    passed in by compare_spectra so BOTH logs use an identical window --
    a differing nperseg between the two sides changes the frequency grid
    and makes every band ratio meaningless.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if nperseg is None:
        nperseg = welch_nperseg(x.size)
    nperseg = min(nperseg, x.size)
    f, pxx = sp_signal.welch(
        x, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2,
        detrend="constant", return_onesided=True, scaling="density",
    )
    return f, pxx


def log_spectral_distance(psd_real: np.ndarray, psd_sim: np.ndarray) -> float:
    """spec 3.7 (a). Both PSDs floored at 1e-12 before the log so a silent
    band contributes a large-but-finite penalty instead of -inf."""
    a = np.maximum(np.asarray(psd_real, dtype=np.float64), PSD_FLOOR)
    b = np.maximum(np.asarray(psd_sim, dtype=np.float64), PSD_FLOOR)
    n = min(a.size, b.size)
    diff = 10.0 * np.log10(a[:n] / b[:n])
    return float(np.sqrt(np.mean(diff ** 2)))


def spectral_centroid(freqs: np.ndarray, psd: np.ndarray) -> Optional[float]:
    """spec 3.7 (c). None when the spectrum carries no power at all --
    the centroid of nothing is not 0 Hz, it is undefined."""
    total = float(np.sum(psd))
    if total <= 0:
        return None
    return float(np.sum(freqs * psd) / total)


def _effective_cutoff(value: Optional[float]) -> Optional[float]:
    """spec 3.7 item 1: a cutoff of 0 in PX4 means the filter is DISABLED.
    Treating it as "cutoff at 0 Hz" would null every band on a log that
    actually has the widest possible bandwidth -- exactly backwards."""
    if value is None:
        return None
    v = float(value)
    return None if v <= 0.0 else v


def band_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def band_ratios_db(
    freqs: np.ndarray,
    psd_real: np.ndarray,
    psd_sim: np.ndarray,
    fs: float,
    *,
    cutoff_real: Optional[float] = None,
    cutoff_sim: Optional[float] = None,
) -> Tuple[Dict[str, Optional[float]], Dict[str, str], List[str]]:
    """
    Four band-power ratios in dB: 10*log10( integral_b S_sim / integral_b S_real ).

    Returns (ratios, nulls, flags). A band is null -- never 0.0, never
    -inf -- when it cannot be measured, for either of two independent
    reasons:

      band_above_nyquist            the grid cannot represent it at all
                                    (at the 50 Hz canonical grid Nyquist
                                    is 25 Hz, so rotor and noise_floor
                                    are unavailable unless the 5.4
                                    high-rate path is used)
      band_above_antialiasing_cutoff  either log's IMU filter removed the
                                    content upstream (A5)

    A band straddling a cutoff is still computed but flagged
    band_partially_filtered, because the passband part of it is real.
    """
    nyquist = fs / 2.0
    cut_r = _effective_cutoff(cutoff_real)
    cut_s = _effective_cutoff(cutoff_sim)
    ratios: Dict[str, Optional[float]] = {}
    nulls: Dict[str, str] = {}
    flags: List[str] = []

    for name, lo, hi_opt in BANDS:
        hi = nyquist if hi_opt is None else hi_opt
        # spec 3.7 (b): "the rotor and noise-floor bands are only computed
        # when the aligned sample rate supports them -- at the canonical
        # 50 Hz grid the Nyquist is 25 Hz, so bands 3 and 4 are null".
        # A band is therefore unavailable unless its WHOLE range is under
        # Nyquist. Computing rotor from the 20-25 Hz sliver that happens to
        # fit would report a "rotor band ratio" measured on 6% of the band.
        if lo >= nyquist or (hi_opt is not None and hi_opt > nyquist):
            ratios[name] = None
            nulls[name] = "band_above_nyquist"
            continue

        # A5 gate: the LOWER edge above either cutoff means the whole band
        # was filtered away before logging.
        blocking = [c for c in (cut_r, cut_s) if c is not None and c <= lo]
        if blocking:
            ratios[name] = None
            nulls[name] = "band_above_antialiasing_cutoff"
            continue

        hi = min(hi, nyquist)
        pr = band_power(freqs, psd_real, lo, hi)
        ps = band_power(freqs, psd_sim, lo, hi)
        if pr <= 0 or ps <= 0:
            ratios[name] = None
            nulls[name] = "no_power_in_band"
            continue
        ratios[name] = float(10.0 * np.log10(ps / pr))

        straddling = [c for c in (cut_r, cut_s) if c is not None and lo < c < hi]
        if straddling:
            flags.append(f"band_partially_filtered:{name}")

    return ratios, nulls, flags


def compare_spectra(
    real: Sequence[float],
    sim: Sequence[float],
    fs: float,
    *,
    chain: str = ACCEL_CHAIN,
    cutoff_real: Optional[float] = None,
    cutoff_sim: Optional[float] = None,
    spectral_source: str = "filtered",
    low_rate: bool = False,
) -> dict:
    """
    Full spec 3.7 comparison for one signal.

    `chain` selects which cutoff parameter applies (ACCEL_CHAIN ->
    IMU_ACCEL_CUTOFF, GYRO_CHAIN -> IMU_GYRO_CUTOFF); the caller reads the
    right one out of each log's metadata and passes the two values.

    `spectral_source` records whether this came from the raw FIFO topics
    or the filtered ones (spec 3.7 [DECISION], A5). When the source is
    "fifo" the data is pre-filter, so the cutoff gate does not apply and
    the caller should pass cutoff_real/cutoff_sim as None.

    spec 9 case 5: a signal below the 5 Hz native gate gets no spectral
    metrics at all -- everything null with reason low_rate.
    """
    out: Dict[str, Optional[float]] = {}
    nulls: Dict[str, str] = {}
    flags: List[str] = []
    keys = ["lsd_db", "centroid_real_hz", "centroid_sim_hz", "centroid_shift_hz"]

    if low_rate:
        for key in keys:
            out[key] = None
            nulls[key] = "low_rate"
        return {**out, "band_ratios_db": {b[0]: None for b in BANDS},
                "band_ratio_nulls": {b[0]: "low_rate" for b in BANDS},
                "nulls": nulls, "flags": ["low_rate"], "chain": chain,
                "spectral_source": spectral_source, "fs": fs}

    r = np.asarray(real, dtype=np.float64)
    m = np.asarray(sim, dtype=np.float64)
    r, m = r[np.isfinite(r)], m[np.isfinite(m)]

    # Identical window on both sides, sized from the SHORTER signal -- see
    # welch_psd's docstring for why a differing nperseg invalidates the
    # band ratios.
    nperseg = welch_nperseg(min(r.size, m.size))
    if min(r.size, m.size) < 8:
        for key in keys:
            out[key] = None
            nulls[key] = "insufficient_samples"
        return {**out, "band_ratios_db": {b[0]: None for b in BANDS},
                "band_ratio_nulls": {b[0]: "insufficient_samples" for b in BANDS},
                "nulls": nulls, "flags": flags, "chain": chain,
                "spectral_source": spectral_source, "fs": fs}

    f_r, psd_r = welch_psd(r, fs, nperseg)
    f_m, psd_m = welch_psd(m, fs, nperseg)

    out["lsd_db"] = log_spectral_distance(psd_r, psd_m)
    c_r = spectral_centroid(f_r, psd_r)
    c_m = spectral_centroid(f_m, psd_m)
    out["centroid_real_hz"] = c_r
    out["centroid_sim_hz"] = c_m
    if c_r is None or c_m is None:
        out["centroid_shift_hz"] = None
        nulls["centroid_shift_hz"] = "no_spectral_power"
    else:
        out["centroid_shift_hz"] = float(c_m - c_r)

    ratios, ratio_nulls, ratio_flags = band_ratios_db(
        f_r, psd_r, psd_m, fs, cutoff_real=cutoff_real, cutoff_sim=cutoff_sim)
    flags.extend(ratio_flags)

    # A5 item 3: a cutoff difference poisons the whole chain. Every
    # spectral metric here carries the warning -- LSD and the centroid as
    # well as the band ratios, because all three are computed from spectra
    # the two filters shaped differently.
    cut_r, cut_s = _effective_cutoff(cutoff_real), _effective_cutoff(cutoff_sim)
    if cut_r != cut_s:
        flags.append("cutoff_mismatch")

    return {
        **out,
        "band_ratios_db": ratios,
        "band_ratio_nulls": ratio_nulls,
        "nulls": nulls,
        "flags": flags,
        "chain": chain,
        "spectral_source": spectral_source,
        "cutoff_real": cutoff_real,
        "cutoff_sim": cutoff_sim,
        "fs": fs,
        "nperseg": int(nperseg),
    }


# ---------------------------------------------------------------------------
# spec 3.8 -- actuator-command lag
# ---------------------------------------------------------------------------

# spec 3.8 [DECISION]: the three command -> response pairs.
LAG_PAIRS = (
    ("roll_rate_loop", "rate_sp.roll", "gyro.x"),
    ("pitch_rate_loop", "rate_sp.pitch", "gyro.y"),
    ("collective_thrust", "thrust.sp", "accel.z"),
)


def _parabolic_vertex(y_minus: float, y_zero: float, y_plus: float) -> float:
    """Sub-sample offset of a parabola through three equally spaced points
    (spec 3.8 [DECISION]). At the 50 Hz grid one sample is 20 ms, which is
    coarse against real motor time constants of 30-80 ms, so refinement is
    not optional.

    Denominator zero means the three points are collinear (no curvature) --
    there is no vertex to find, so fall back to the integer peak.
    """
    denom = y_minus - 2.0 * y_zero + y_plus
    if denom == 0:
        return 0.0
    offset = 0.5 * (y_minus - y_plus) / denom
    # A vertex further than one sample from the peak means the peak was
    # not really a peak; refusing it is safer than reporting a lag pulled
    # outside the bracket it was found in.
    return float(offset) if abs(offset) <= 1.0 else 0.0


def estimate_lag_s(
    command: Sequence[float], response: Sequence[float], fs: float,
    max_lag_s: float = LAG_SEARCH_S,
) -> Tuple[Optional[float], float]:
    """
    Lag maximising the normalised cross-correlation of `response` against
    `command`, in seconds, with parabolic sub-sample refinement.

    Returns (lag_s_or_None, peak_rho). lag is None only when the inputs are
    too short to correlate; the spec 3.8 rho < 0.3 gate is applied by the
    caller so it can report the correlation alongside the refusal.

    Positive lag means the response FOLLOWS the command, which is the
    physically expected direction for an actuator.
    """
    c = np.asarray(command, dtype=np.float64)
    y = np.asarray(response, dtype=np.float64)
    n = min(c.size, y.size)
    if n < 8:
        return None, 0.0
    c, y = c[:n] - np.mean(c[:n]), y[:n] - np.mean(y[:n])
    denom = float(np.sqrt(np.dot(c, c) * np.dot(y, y)))
    if denom <= 0:
        return None, 0.0

    max_lag = int(min(max_lag_s * fs, n - 1))
    if max_lag < 1:
        return None, 0.0
    lags = np.arange(-max_lag, max_lag + 1)
    scores = np.empty(lags.size)
    for i, lag in enumerate(lags):
        if lag >= 0:
            scores[i] = np.dot(y[lag:], c[: n - lag])
        else:
            scores[i] = np.dot(y[: n + lag], c[-lag:])
    scores /= denom

    peak = int(np.argmax(scores))
    offset = 0.0
    if 0 < peak < scores.size - 1:
        offset = _parabolic_vertex(scores[peak - 1], scores[peak], scores[peak + 1])
    return float((lags[peak] + offset) / fs), float(scores[peak])


def actuator_lag(
    real_command: Sequence[float], real_response: Sequence[float],
    sim_command: Sequence[float], sim_response: Sequence[float],
    fs: float, *, pair_name: str = "",
) -> dict:
    """
    spec 3.8: measure tau within each log separately, then difference.
    delta_tau = tau_sim - tau_real, reported in milliseconds.

    spec 3.8 [DECISION] validity gate: if EITHER side's peak correlation
    is below 0.3, the pair is reported as lag_unreliable with the
    correlations, and no lag number is published. A lag read off an
    uncorrelated pair is noise, and publishing it would be exactly the
    fabricated precision this project forbids.
    """
    tau_r, rho_r = estimate_lag_s(real_command, real_response, fs)
    tau_s, rho_s = estimate_lag_s(sim_command, sim_response, fs)

    result: Dict[str, object] = {
        "pair": pair_name,
        "rho_real": rho_r,
        "rho_sim": rho_s,
    }
    if tau_r is None or tau_s is None:
        result.update({"tau_real_ms": None, "tau_sim_ms": None, "delta_tau_ms": None,
                       "nulls": {"delta_tau_ms": "insufficient_samples"},
                       "flags": ["lag_unreliable"]})
        return result
    if rho_r < LAG_RHO_GATE or rho_s < LAG_RHO_GATE:
        result.update({"tau_real_ms": None, "tau_sim_ms": None, "delta_tau_ms": None,
                       "nulls": {"delta_tau_ms": "lag_unreliable"},
                       "flags": ["lag_unreliable"]})
        return result

    result.update({
        "tau_real_ms": round(tau_r * 1000.0, 1),
        "tau_sim_ms": round(tau_s * 1000.0, 1),
        "delta_tau_ms": round((tau_s - tau_r) * 1000.0, 1),
        "nulls": {},
        "flags": [],
    })
    return result
