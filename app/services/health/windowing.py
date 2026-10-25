"""
Single-flight windowing (health spec 1.2, build step 2).

One log, no partner: find when the drone was airborne, which stretches of
that were "steady" (straight and level enough for steady-state features),
how long each flight phase lasted, and whether the flight qualifies at all.

Reuses reality_gap.align for the parts that already existed (the arm anchor,
the altitude-based airborne window, phase classification, resampling).
The one addition is preferring PX4's own land detector over the altitude
threshold when the log carries it: alt.rel is relative to the EKF origin and
drifts with the baro, while vehicle_land_detected.landed is the autopilot's
own decision about whether it is on the ground.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.services.reality_gap.align import (
    CANONICAL_RATE_HZ,
    PHASE_VZ_THRESHOLD,
    airborne_window,
    classify_phases,
    find_arm_anchor,
    resample_to_grid,
)
from app.services.reality_gap.parsers import ParsedLog

MIN_AIRBORNE_S = 60.0       # spec 1.2: a flight qualifies with >= 60 s airborne
MIN_STEADY_S = 30.0         # spec 1.2: steady-only features need >= 30 s steady
STEADY_MIN_SEGMENT_S = 5.0  # spec 1.2: a steady segment is at least 5 s long
STEADY_VZ_MPS = PHASE_VZ_THRESHOLD  # |vz| < 0.5 m/s, the existing phase threshold
STEADY_RATE_RADPS = 0.3     # spec 1.2: |rate setpoint| < 0.3 rad/s on every axis
RATE_SP_KEYS = ("rate_sp.roll", "rate_sp.pitch", "rate_sp.yaw")


@dataclass
class FlightWindow:
    qualified: bool
    reasons: List[str]                       # why it didn't qualify; empty if it did
    airborne_start_us: Optional[int] = None
    airborne_end_us: Optional[int] = None
    airborne_s: float = 0.0
    airborne_source: str = "none"            # land_detector | altitude_threshold | none
    takeoff_us: Optional[int] = None
    steady_segments: List[Tuple[int, int]] = field(default_factory=list)  # (start_us, end_us)
    steady_s: float = 0.0
    steady_qualified: bool = False
    steady_reason: Optional[str] = None      # why steady detection couldn't run, if it couldn't
    phase_seconds: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """(start, end_exclusive) index pairs of every run of True."""
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def _airborne_from_land_detector(parsed: ParsedLog) -> Optional[Tuple[int, int, int]]:
    """Longest contiguous run of landed == 0, as (start_us, end_us, n_runs).
    The landed flag is a state that holds until it changes, so each run ends
    at the first sample that says landed again (or the last sample)."""
    sig = parsed.signals.get("land.landed")
    if sig is None or sig.values.size < 2:
        return None
    airborne = sig.values < 0.5
    runs = _runs(airborne)
    if not runs:
        return None
    best = max(runs, key=lambda r: sig.t_us[min(r[1], sig.t_us.size - 1)] - sig.t_us[r[0]])
    start = int(sig.t_us[best[0]])
    end = int(sig.t_us[min(best[1], sig.t_us.size - 1)])
    return start, end, len(runs)


def window_flight(parsed: ParsedLog) -> FlightWindow:
    warnings: List[str] = []
    reasons: List[str] = []

    source = "none"
    start_us = end_us = None
    land = _airborne_from_land_detector(parsed)
    if land is not None:
        start_us, end_us, n_runs = land
        source = "land_detector"
        if n_runs > 1:
            warnings.append(f"multiple_airborne_segments:{n_runs}; longest used")
    else:
        anchor, _ = find_arm_anchor(parsed)
        if anchor is not None:
            s, e, w = airborne_window(parsed, anchor)
            warnings.extend(w)
            if "ground_only" not in w and e is not None:
                start_us, end_us, source = s, e, "altitude_threshold"

    if start_us is None or end_us is None or end_us <= start_us:
        return FlightWindow(qualified=False, reasons=["no_airborne_detected"], warnings=warnings)

    airborne_s = (end_us - start_us) / 1e6
    if airborne_s < MIN_AIRBORNE_S:
        reasons.append(f"airborne_too_short:{airborne_s:.1f}s<{MIN_AIRBORNE_S:.0f}s")

    win = FlightWindow(
        qualified=not reasons,
        reasons=reasons,
        airborne_start_us=int(start_us),
        airborne_end_us=int(end_us),
        airborne_s=round(airborne_s, 2),
        airborne_source=source,
        takeoff_us=int(start_us),
        warnings=warnings,
    )

    vz = parsed.signals.get("vel.z")
    missing = [k for k in ("vel.z",) + RATE_SP_KEYS if k not in parsed.signals]
    if missing:
        win.steady_reason = "signal_absent:" + ",".join(missing)
        return win

    grid, vz_g = resample_to_grid(vz, start_us, end_us, CANONICAL_RATE_HZ, zero_order_hold=False)
    with np.errstate(invalid="ignore"):
        steady = np.isfinite(vz_g) & (np.abs(vz_g) < STEADY_VZ_MPS)
        for key in RATE_SP_KEYS:
            _, r = resample_to_grid(parsed.signals[key], start_us, end_us, CANONICAL_RATE_HZ, zero_order_hold=False)
            steady &= np.isfinite(r) & (np.abs(r) < STEADY_RATE_RADPS)

    min_len = int(round(STEADY_MIN_SEGMENT_S * CANONICAL_RATE_HZ))
    dt = 1.0 / CANONICAL_RATE_HZ
    segments = []
    for a, b in _runs(steady):
        if b - a >= min_len:
            segments.append((int(grid[a]), int(grid[b - 1])))
    win.steady_segments = segments
    win.steady_s = round(sum((e - s) / 1e6 + dt for s, e in segments), 2)
    win.steady_qualified = win.qualified and win.steady_s >= MIN_STEADY_S
    if win.qualified and not win.steady_qualified:
        win.steady_reason = f"insufficient_steady_flight:{win.steady_s:.1f}s<{MIN_STEADY_S:.0f}s"

    win.phase_seconds = {name: round(int(np.nansum(m)) * dt, 2) for name, m in classify_phases(vz_g).items()}
    return win
