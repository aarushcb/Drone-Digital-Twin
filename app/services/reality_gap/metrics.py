"""
D1 spec 3.1-3.6 and 3.9 -- per-signal comparison metrics.

WHAT RUNS HERE AND ON WHAT DATA (spec 3.0 [DECISION], amendment A7):
everything in this file runs on the 50 Hz resampled grid, where one
signal over a 20-minute flight is 60,000 samples ~= 240 KB as float32 and
the entire 25-signal grid for both logs is ~12 MB. That is not a
constraint on a 512 MB tier, so every metric here is computed EXACTLY:
scipy.stats.ks_2samp and scipy.stats.wasserstein_distance are called
directly on the samples, and Freedman-Diaconis binning uses a full-sort
exact IQR. No streaming quantiles, no reservoir sampling, no
histogram-derived KS surrogate.

That is a deliberate instruction, not an oversight. Spec 5.7's memory
discipline governs PARSING raw high-rate topics, where one log holds
millions of samples; it does not govern metric computation, and the two
are separated by the 5.4 resample. Substituting an approximation here
would weaken a headline metric and break spec 10.1 item 7 ("must match
scipy exactly") to solve a problem that does not exist.

NULL IS A VALUE, NOT A ZERO (spec 9, and 10.6 item 29): where the spec
requires a metric to be unavailable, it is returned as None with a
machine-readable reason in the result's `nulls` map. Nothing here ever
coerces an unavailable metric to 0.0 -- a fabricated zero reads as
"perfect agreement", which is the exact opposite of "we could not
measure this".
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sp_stats

# --- spec 3.0 constants ---
EPSILON = 1e-6          # Laplace smoothing, spec 3.0 [DECISION]
MIN_BINS, MAX_BINS = 20, 200
RANGE_FLOOR = 1e-9      # spec 3.1: below this, NRMSE_range is null
DEGENERATE_VAR = 1e-12  # spec 3.2 / 9 case 10: below this, the signal is degenerate

LN2 = np.log(2.0)


class _Result:
    """Accumulator so every metric writes its value and, when unavailable,
    its reason in the same place. Plain dicts on the way out -- the rest of
    this codebase's services return dicts, and these end up as JSON."""

    def __init__(self) -> None:
        self.values: Dict[str, Optional[float]] = {}
        self.nulls: Dict[str, str] = {}
        self.flags: List[str] = []

    def set(self, key: str, value: float) -> None:
        self.values[key] = float(value)

    def null(self, key: str, reason: str) -> None:
        self.values[key] = None
        self.nulls[key] = reason

    def flag(self, name: str) -> None:
        if name not in self.flags:
            self.flags.append(name)

    def as_dict(self) -> dict:
        return {**self.values, "nulls": self.nulls, "flags": self.flags}


# ---------------------------------------------------------------------------
# spec 3.0 -- shared binning
# ---------------------------------------------------------------------------


def freedman_diaconis_edges(pooled: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Common bin edges for both histograms, from the POOLED real+sim sample
    (spec 3.0 [DECISION]): h = 2*IQR*n^(-1/3), bin count clamped to
    [20, 200]; IQR == 0 falls back to sqrt(n) bins over the pooled range;
    pooled range == 0 is degenerate and the caller nulls everything.

    The IQR is exact -- np.percentile sorts. Spec 3.0's A7 decision
    requires that explicitly: an approximate quantile here would change
    the bin edges, which changes KL/PSI/JSD, which is precisely the
    "silently distorting the number" failure spec 10.1 item 6 exists to
    catch.

    Why a common edge set at all: KL and PSI compare bin-for-bin, so the
    two histograms must be on identical edges or the comparison is
    meaningless.
    """
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return None, "no_finite_samples"
    lo, hi = float(pooled.min()), float(pooled.max())
    span = hi - lo
    if span <= 0:
        return None, "degenerate"

    q75, q25 = np.percentile(pooled, [75, 25])
    iqr = float(q75 - q25)
    if iqr > 0:
        width = 2.0 * iqr * pooled.size ** (-1.0 / 3.0)
        n_bins = int(np.ceil(span / width)) if width > 0 else MIN_BINS
    else:
        # Constant-ish but not constant (e.g. a signal that sits at one
        # value except for a couple of spikes): FD's width collapses to 0,
        # so fall back to sqrt(n) rather than dividing by zero.
        n_bins = int(np.ceil(np.sqrt(pooled.size)))
    n_bins = int(np.clip(n_bins, MIN_BINS, MAX_BINS))
    return np.linspace(lo, hi, n_bins + 1), None


def _smoothed_histograms(
    real: np.ndarray, sim: np.ndarray, edges: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalised histograms with add-epsilon smoothing (spec 3.0
    [DECISION]). Smoothing is applied BEFORE normalisation so both
    distributions still sum to 1; without it KL is infinite wherever the
    sim never visited a bin the real flight did, which is constantly."""
    p, _ = np.histogram(real, bins=edges)
    q, _ = np.histogram(sim, bins=edges)
    p = p.astype(np.float64) + EPSILON
    q = q.astype(np.float64) + EPSILON
    return p / p.sum(), q / q.sum()


def kl_divergence_bits(p: np.ndarray, q: np.ndarray) -> float:
    """D_KL(P||Q) in BITS. Direction is fixed by spec 3.3 [DECISION]:
    P = real, Q = sim. KL is asymmetric and reporting it without the
    direction is a bug, so callers must not swap the arguments."""
    return float(np.sum(p * np.log2(p / q)))


def jensen_shannon_bits(p: np.ndarray, q: np.ndarray) -> float:
    """Symmetric, bounded on [0, 1] in bits (spec 3.3). This is what the
    spec 4 scoring function consumes, because a bounded quantity maps
    cleanly onto a 0-100 score; raw KL does not."""
    m = 0.5 * (p + q)
    return float(0.5 * kl_divergence_bits(p, m) + 0.5 * kl_divergence_bits(q, m))


def population_stability_index(p: np.ndarray, q: np.ndarray) -> float:
    """PSI, spec 3.4. NATURAL log, unlike KL/JSD above which are in bits --
    that is the convention the 0.1 / 0.25 industry thresholds are quoted
    against, and changing the base would silently move those bands."""
    return float(np.sum((p - q) * np.log(p / q)))


# ---------------------------------------------------------------------------
# spec 3.6 -- effective sample size
# ---------------------------------------------------------------------------


def effective_sample_size(x: np.ndarray) -> Optional[float]:
    """
    N_eff = N * (1 - phi) / (1 + phi), phi = lag-1 autocorrelation
    (spec 3.6 [DECISION]).

    APPROXIMATION, flagged as the spec requires: this is exact only for a
    true AR(1) process. Flight telemetry is not AR(1) -- it has structure
    at many lags -- so N_eff is an indicative correction to the nominal N,
    not a rigorous independent-sample count. It exists to stop a reader
    treating a p-value computed from 60,000 heavily autocorrelated
    samples as if it came from 60,000 independent draws.

    phi >= 1 (a perfectly correlated or constant series) would give
    N_eff <= 0; clamped to None rather than returning a nonsense number.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3:
        return None
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom <= 0:
        return None
    phi = float(np.dot(xc[:-1], xc[1:]) / denom)
    if phi >= 1.0 - 1e-12:
        return None
    return float(x.size * (1.0 - phi) / (1.0 + phi))


# ---------------------------------------------------------------------------
# spec 3.1-3.6 -- the per-signal comparison
# ---------------------------------------------------------------------------


def compare_signal(
    real: Sequence[float],
    sim: Sequence[float],
    *,
    low_rate: bool = False,
) -> dict:
    """
    Every spec 3.1-3.6 metric for one signal, both sides already aligned
    and on the common 50 Hz grid.

    `low_rate` marks a signal below the 5 Hz native gate (spec 9 case 5):
    point-wise metrics are still computed and the flag is attached; the
    caller is responsible for nulling the SPECTRAL metrics, which live in
    spectral.py.

    Returns a flat dict of metric -> value (None where unavailable), plus
    `nulls` mapping each unavailable metric to its reason and `flags` for
    advisory conditions.
    """
    out = _Result()
    r = np.asarray(real, dtype=np.float64)
    m = np.asarray(sim, dtype=np.float64)

    # Point-wise metrics require paired samples; the grid guarantees equal
    # length, but a caller passing ragged input should not get a silently
    # truncated comparison without being told.
    n = min(r.size, m.size)
    if n != r.size or n != m.size:
        out.flag("length_mismatch_truncated")
    r, m = r[:n], m[:n]

    pair_finite = np.isfinite(r) & np.isfinite(m)
    if int(pair_finite.sum()) < 2:
        for key in ("nrmse_range", "nrmse_sigma", "bias", "pearson_r", "vr", "vr_log",
                    "kl_bits_real_vs_sim", "jsd_bits", "psi", "wasserstein",
                    "wasserstein_normalised", "ks_statistic", "ks_pvalue", "n_eff"):
            out.null(key, "insufficient_paired_samples")
        out.values["n_samples"] = int(n)
        return out.as_dict()

    rp, mp = r[pair_finite], m[pair_finite]
    out.values["n_samples"] = int(rp.size)
    if low_rate:
        out.flag("low_rate")

    var_r = float(np.var(rp, ddof=1)) if rp.size > 1 else 0.0
    var_m = float(np.var(mp, ddof=1)) if mp.size > 1 else 0.0
    range_r = float(rp.max() - rp.min())

    # spec 9 case 10: a stuck sensor on EITHER side is degenerate. It is a
    # real finding (a frozen real sensor is worth seeing), not an error.
    degenerate = var_r < DEGENERATE_VAR or var_m < DEGENERATE_VAR
    if degenerate:
        out.flag("degenerate")

    # --- 3.1 NRMSE, bias, Pearson ---
    rmse = float(np.sqrt(np.mean((mp - rp) ** 2)))
    out.set("rmse", rmse)
    out.set("bias", float(mp.mean() - rp.mean()))

    if range_r < RANGE_FLOOR:
        out.null("nrmse_range", "range_below_floor")
    else:
        out.set("nrmse_range", rmse / range_r)

    # spec 9 case 10: NRMSE_sigma is the fallback, but it divides by the
    # REAL side's sigma -- so it survives a degenerate SIM and dies with a
    # degenerate REAL.
    if var_r < DEGENERATE_VAR:
        out.null("nrmse_sigma", "real_variance_degenerate")
    else:
        out.set("nrmse_sigma", rmse / float(np.sqrt(var_r)))

    if var_r < DEGENERATE_VAR or var_m < DEGENERATE_VAR:
        out.null("pearson_r", "degenerate_variance")
    else:
        out.set("pearson_r", float(np.corrcoef(rp, mp)[0, 1]))

    # Which normalisation is primary, per spec 3.1 [DECISION].
    out.values["nrmse_primary"] = (
        "nrmse_sigma" if out.values.get("nrmse_range") is None else "nrmse_range"
    )

    # --- 3.2 variance ratio ---
    # spec 3.2 nulls VR on a degenerate REAL side; spec 9 case 10 broadens
    # that to a zero-variance signal on EITHER side. Both matter: a stuck
    # SIM gives VR = 0 and VR_log = log2(0) = -inf, and reporting 0.0 would
    # read as "the simulator is infinitely too smooth" rather than "this
    # sensor was stuck".
    if degenerate:
        reason = ("real_variance_degenerate" if var_r < DEGENERATE_VAR
                  else "sim_variance_degenerate")
        out.null("vr", reason)
        out.null("vr_log", reason)
    else:
        vr = var_m / var_r
        out.set("vr", vr)
        if vr <= 0:
            out.null("vr_log", "sim_variance_zero")
        else:
            out.set("vr_log", float(np.log2(vr)))

    # --- 3.3-3.4 distribution metrics over common bins ---
    edges, edge_reason = freedman_diaconis_edges(np.concatenate([rp, mp]))
    if edges is None or degenerate:
        reason = edge_reason or "degenerate"
        for key in ("kl_bits_real_vs_sim", "jsd_bits", "psi"):
            out.null(key, reason)
        out.values["n_bins"] = None
    else:
        p, q = _smoothed_histograms(rp, mp, edges)
        out.values["n_bins"] = int(edges.size - 1)
        out.set("kl_bits_real_vs_sim", kl_divergence_bits(p, q))
        out.set("jsd_bits", jensen_shannon_bits(p, q))
        out.set("psi", population_stability_index(p, q))

    # --- 3.5 Wasserstein-1, exact, on raw samples ---
    if degenerate:
        out.null("wasserstein", "degenerate")
        out.null("wasserstein_normalised", "degenerate")
    else:
        w1 = float(sp_stats.wasserstein_distance(rp, mp))
        out.set("wasserstein", w1)
        if range_r < RANGE_FLOOR:
            out.null("wasserstein_normalised", "range_below_floor")
        else:
            out.set("wasserstein_normalised", w1 / range_r)

    # --- 3.6 KS, exact, via scipy ---
    if degenerate:
        out.null("ks_statistic", "degenerate")
        out.null("ks_pvalue", "degenerate")
    else:
        ks = sp_stats.ks_2samp(rp, mp)
        out.set("ks_statistic", float(ks.statistic))
        out.set("ks_pvalue", float(ks.pvalue))

    n_eff = effective_sample_size(rp)
    if n_eff is None:
        out.null("n_eff", "autocorrelation_undefined")
    else:
        out.set("n_eff", n_eff)

    # spec 3.6 [DECISION]: the p-value must never be read as significance
    # at these sample sizes. Carried as a flag so the UI cannot forget.
    out.flag("pvalue_not_meaningful_at_this_n")

    return out.as_dict()


# ---------------------------------------------------------------------------
# spec 3.9 -- resource usage
# ---------------------------------------------------------------------------


def resource_panel(
    real_cpu: Optional[Sequence[float]] = None,
    sim_cpu: Optional[Sequence[float]] = None,
    real_mem: Optional[Sequence[float]] = None,
    sim_mem: Optional[Sequence[float]] = None,
) -> dict:
    """
    spec 3.9 -- mean and 95th-percentile CPU load and RAM for both logs.

    NOT SCORED, and the return value says so in `scored: False`. A SITL
    run's CPU load is a property of the developer's laptop, not of the
    airframe, so folding it into a fidelity score would be meaningless.
    It is reported because high real-side CPU load is a legitimate
    explanation for timing divergence measured elsewhere.
    """
    out: Dict[str, Optional[float]] = {}
    nulls: Dict[str, str] = {}
    for name, series in (("cpu_real", real_cpu), ("cpu_sim", sim_cpu),
                         ("mem_real", real_mem), ("mem_sim", sim_mem)):
        arr = np.asarray(series, dtype=np.float64) if series is not None else np.empty(0)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            out[f"{name}_mean"] = None
            out[f"{name}_p95"] = None
            nulls[f"{name}_mean"] = "topic_absent"
            nulls[f"{name}_p95"] = "topic_absent"
        else:
            out[f"{name}_mean"] = float(arr.mean())
            out[f"{name}_p95"] = float(np.percentile(arr, 95))
    return {**out, "nulls": nulls, "scored": False}
