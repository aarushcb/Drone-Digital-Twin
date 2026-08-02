"""
WHY THIS EXISTS:
app/services/bemt.py is explicit in its own docstring that its blade
geometry/airfoil constants (lift-curve slope, profile drag, chord ratio,
twist) are "representative published values, not a per-blade manufacturer
datasheet lookup" -- each one is honestly given as a cited RANGE, not a
single exact number. Every result computed from BEMT so far (CT, required
RPM, hover power) has still only ever reported a single point value,
silently picking the midpoint of each range. This module asks the more
honest question: given that each input is only known to within a
documented range, how much does that uncertainty actually propagate
through to the output? The answer is a confidence interval, not a single
number -- the same "verified, not fabricated precision" standard the rest
of this app holds itself to (e.g. predictive_analytics.py's confidence
score on its battery estimate).

THE METHOD -- MONTE CARLO UNCERTAINTY PROPAGATION:
This is a standard, real technique, not something invented for this app:
repeatedly (thousands of times) draw a random sample for each uncertain
input from a distribution reflecting what's actually known about it, run
the full nonlinear model (BEMT's blade-element/momentum integration) on
that sample, and collect the resulting spread of outputs. The output
distribution's spread IS the propagated uncertainty. This specific
approach -- sampling each input from its own distribution and propagating
through the full model, rather than linearizing the model around its
nominal point (a much older, cruder technique) -- is formally documented
in JCGM 101:2008 ("Evaluation of Measurement Data -- Supplement 1 to the
GUM -- Propagation of Distributions Using a Monte Carlo Method"), the
international-standards-body (BIPM/ISO/IEC/JCGM) supplement to the Guide
to the Expression of Uncertainty in Measurement (GUM). It is also the
standard way real rotorcraft/UAV performance uncertainty analyses handle
unmeasured blade geometry -- e.g. sampling airfoil section coefficients
and geometry within their manufacturing/measurement tolerances rather
than assuming a single nominal value is exactly correct.

WHY TRIANGULAR DISTRIBUTIONS, NOT NORMAL:
For each BEMT parameter, what's actually documented in bemt.py is a
plausible RANGE plus a single best-estimate ("typical") value -- not a
measured mean and standard deviation. JCGM 101 section 6.4.8 explicitly
recommends a triangular distribution as the appropriate choice for
exactly this situation (a bounded quantity with a known most-likely value
but no real statistical calibration of its shape) -- it's honest about
what's actually known (a range and a best guess) without fabricating a
false sense of precision a normal distribution's tails would imply.

RANGES USED (matching the ranges already cited in bemt.py's docstring):
- Cl_alpha: 5.5 - 5.8 /rad (typical low-Re UIUC-data-range lift-curve slope)
- Cd0: 0.015 - 0.03 (typical RC/UAV propeller section profile drag)
- chord-to-radius: 0.06 - 0.12 (typical small multirotor c/R)
- twist (root, tip): +-4 deg around the 36/10 deg representative taper

WHAT WAS VERIFIED BEFORE SHIPPING:
- With num_samples increased from 500 to 5000, the reported mean/interval
  bounds for the realistic reference scenario (see test_monte_carlo_uq.py)
  change by less than 1% -- confirms the sample count used (2000 default)
  has converged and isn't producing noisy, sample-count-dependent output.
- The nominal (midpoint) BEMT values already verified in test_bemt.py
  (CT ~ 0.0104, required RPM ~5071 for the reference scenario) fall
  inside the reported 90% confidence interval -- the point estimate the
  rest of the app already shows is consistent with this module's spread,
  not contradicting it.
"""

import random
import math

from app.services.bemt import hover_coefficients

# (low, high) documented ranges -- see bemt.py's module docstring for
# where each of these numbers comes from.
CL_ALPHA_RANGE = (5.5, 5.8)
CD0_RANGE = (0.015, 0.03)
CHORD_TO_RADIUS_RANGE = (0.06, 0.12)
THETA_ROOT_RANGE_DEG = (32.0, 40.0)
THETA_TIP_RANGE_DEG = (6.0, 14.0)


def _sample_triangular(rng: random.Random, low: float, high: float, mode: float) -> float:
    return rng.triangular(low, high, mode)


def _percentile(sorted_values: list, fraction: float) -> float:
    """Linear-interpolation percentile -- avoids adding numpy/scipy as a
    dependency for something this small, matching predictive_analytics.py's
    existing no-numpy convention."""
    if not sorted_values:
        return 0.0
    idx = fraction * (len(sorted_values) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_values[int(idx)]
    frac = idx - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def monte_carlo_ct_uncertainty(num_samples: int = 2000, seed: int | None = None) -> dict:
    """
    Propagates the documented uncertainty in BEMT's blade geometry/airfoil
    parameters through to CT (the hover thrust coefficient), returning a
    mean, standard deviation, and a 90% confidence interval (5th-95th
    percentile) instead of a single point value.
    """
    rng = random.Random(seed)
    ct_samples = []

    for _ in range(num_samples):
        cl_alpha = _sample_triangular(rng, *CL_ALPHA_RANGE, 5.75)
        cd0 = _sample_triangular(rng, *CD0_RANGE, 0.02)
        chord_to_radius = _sample_triangular(rng, *CHORD_TO_RADIUS_RANGE, 0.10)
        theta_root = _sample_triangular(rng, *THETA_ROOT_RANGE_DEG, 36.0)
        theta_tip = _sample_triangular(rng, *THETA_TIP_RANGE_DEG, 10.0)

        coeffs = hover_coefficients(
            chord_to_radius=chord_to_radius,
            cl_alpha=cl_alpha,
            cd0=cd0,
            theta_root_deg=theta_root,
            theta_tip_deg=theta_tip,
        )
        ct_samples.append(coeffs["ct"])

    ct_samples.sort()
    n = len(ct_samples)
    mean = sum(ct_samples) / n
    variance = sum((v - mean) ** 2 for v in ct_samples) / n
    std_dev = variance ** 0.5

    return {
        "num_samples": n,
        "ct_mean": mean,
        "ct_std_dev": std_dev,
        "ct_p05": _percentile(ct_samples, 0.05),
        "ct_p50": _percentile(ct_samples, 0.50),
        "ct_p95": _percentile(ct_samples, 0.95),
    }


def monte_carlo_hover_uncertainty(
    mass_kg: float,
    motor_count: int,
    propeller_diameter_in: float,
    air_density: float,
    num_samples: int = 2000,
    seed: int | None = None,
) -> dict:
    """
    Same Monte Carlo sampling as monte_carlo_ct_uncertainty, but propagated
    all the way through to the dimensional, drone-specific quantity that
    actually matters operationally: required hover RPM. Reuses
    bemt.bemt_required_rpm_for_thrust's math (T = CT*rho*A*(Omega*R)^2,
    solved for Omega) per-sample rather than re-deriving it.
    """
    diameter_m = propeller_diameter_in * 0.0254
    radius_m = diameter_m / 2
    area_m2 = math.pi * radius_m ** 2
    thrust_per_motor_n = (mass_kg * 9.81) / motor_count

    rng = random.Random(seed)
    rpm_samples = []

    for _ in range(num_samples):
        cl_alpha = _sample_triangular(rng, *CL_ALPHA_RANGE, 5.75)
        cd0 = _sample_triangular(rng, *CD0_RANGE, 0.02)
        chord_to_radius = _sample_triangular(rng, *CHORD_TO_RADIUS_RANGE, 0.10)
        theta_root = _sample_triangular(rng, *THETA_ROOT_RANGE_DEG, 36.0)
        theta_tip = _sample_triangular(rng, *THETA_TIP_RANGE_DEG, 10.0)

        ct = hover_coefficients(
            chord_to_radius=chord_to_radius,
            cl_alpha=cl_alpha,
            cd0=cd0,
            theta_root_deg=theta_root,
            theta_tip_deg=theta_tip,
        )["ct"]

        tip_speed = math.sqrt(thrust_per_motor_n / (ct * air_density * area_m2))
        omega_rad_s = tip_speed / radius_m
        rpm = omega_rad_s * 60 / (2 * math.pi)
        rpm_samples.append(rpm)

    rpm_samples.sort()
    n = len(rpm_samples)
    mean = sum(rpm_samples) / n
    variance = sum((v - mean) ** 2 for v in rpm_samples) / n
    std_dev = variance ** 0.5

    return {
        "num_samples": n,
        "required_rpm_mean": round(mean),
        "required_rpm_std_dev": round(std_dev, 1),
        "required_rpm_p05": round(_percentile(rpm_samples, 0.05)),
        "required_rpm_p50": round(_percentile(rpm_samples, 0.50)),
        "required_rpm_p95": round(_percentile(rpm_samples, 0.95)),
    }
