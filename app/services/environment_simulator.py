"""
WHY THIS EXISTS:
Answers "how would this specific drone perform if flown at a different
altitude/temperature" -- using real, citable physics rather than an
opaque black-box guess.

WHERE EACH NUMBER COMES FROM (so this stays honest about what's real
physics vs. what's an empirical rule-of-thumb):

1. AIR DENSITY -- International Standard Atmosphere (ISA) barometric
   formula for pressure at altitude, combined with the ideal gas law
   using the user's chosen temperature (rather than the standard lapse
   rate temperature, since the whole point is simulating NON-standard
   conditions). Verified against known ISA reference values before
   shipping: sea level/15C -> 1.2250 kg/m^3 (matches the textbook 1.225
   exactly), 3000m -> 0.9091 kg/m^3 (matches the known ~0.909 reference).

2. POWER REQUIRED FOR HOVER vs. AIR DENSITY -- actuator disk (momentum)
   theory, the standard model used in real drone thrust/power
   calculators: for a FIXED thrust requirement (hovering always needs
   thrust = weight, regardless of air density), required power scales
   as 1/sqrt(air density). Thinner air needs more power for the same
   lift. This ratio-based approach doesn't require knowing propeller
   size (which isn't in our stored specs) -- it cancels out when
   comparing two conditions to each other.

3. BATTERY COLD-TEMPERATURE CAPACITY DERATING -- an empirical curve
   built from multiple independent, converging real-world sources
   (Battery University, large-battery.com, ufinebattery.com): roughly
   100% capacity at 25C, ~75% at 0C (sources cite 20-30% loss), ~50% at
   -20C (consistently cited across sources). This is a real documented
   phenomenon, not a made-up number, though the exact curve varies by
   battery chemistry/brand -- treated here as a reasonable estimate,
   not a manufacturer-specific measurement.
"""

import math

# ISA (International Standard Atmosphere) constants
_P0 = 101325       # Pa, sea-level standard pressure
_T0 = 288.15       # K, sea-level standard temperature (15C)
_LAPSE_RATE = 0.0065  # K/m
_GRAVITY = 9.80665    # m/s^2
_MOLAR_MASS_AIR = 0.0289644  # kg/mol
_GAS_CONSTANT = 8.3144598    # J/(mol*K)
_R_SPECIFIC = _GAS_CONSTANT / _MOLAR_MASS_AIR  # ~287.058 J/(kg*K)

SEA_LEVEL_DENSITY = _P0 / (_R_SPECIFIC * _T0)  # ~1.225 kg/m^3, the baseline everything compares against


def air_density(altitude_m: float, temperature_c: float) -> float:
    temp_k = temperature_c + 273.15
    pressure = _P0 * (1 - _LAPSE_RATE * altitude_m / _T0) ** (_GRAVITY * _MOLAR_MASS_AIR / (_GAS_CONSTANT * _LAPSE_RATE))
    return pressure / (_R_SPECIFIC * temp_k)


def hover_power_scale_factor(altitude_m: float, temperature_c: float) -> float:
    """
    How much MORE power is needed to hover at these conditions vs. sea
    level/15C -- 1.0 means no change, 1.2 means 20% more power needed.
    """
    density = air_density(altitude_m, temperature_c)
    return math.sqrt(SEA_LEVEL_DENSITY / density)


# Piecewise-linear interpolation between real cited reference points --
# see module docstring for sources. Battery is assumed to perform at its
# full rated capacity between 15C and 35C (the commonly-cited "sweet
# spot" range), with the documented cold-derating curve applying below
# that.
_BATTERY_TEMP_CAPACITY_POINTS = [
    (-20, 0.50),
    (-10, 0.70),
    (0, 0.75),
    (15, 1.00),
    (35, 1.00),
]


def battery_capacity_factor(temperature_c: float) -> float:
    points = _BATTERY_TEMP_CAPACITY_POINTS
    if temperature_c <= points[0][0]:
        return points[0][1]
    if temperature_c >= points[-1][0]:
        return points[-1][1]
    for i in range(len(points) - 1):
        t0, f0 = points[i]
        t1, f1 = points[i + 1]
        if t0 <= temperature_c <= t1:
            fraction = (temperature_c - t0) / (t1 - t0)
            return f0 + fraction * (f1 - f0)
    return 1.0  # unreachable given the bounds checks above, but a safe fallback


def simulate_conditions(
    altitude_m: float,
    temperature_c: float,
    baseline_drain_rate_percent_per_min: float | None,
) -> dict:
    """
    Combines the above into a single result. If a real empirically-
    observed drain rate is available (from the drone's own Phase 9
    predictive analytics), projects how it would change under these
    conditions -- otherwise just reports the raw physics factors without
    a projected number, rather than fabricating a baseline.
    """
    power_scale = hover_power_scale_factor(altitude_m, temperature_c)
    capacity_factor = battery_capacity_factor(temperature_c)

    projected_drain_rate = None
    if baseline_drain_rate_percent_per_min is not None and baseline_drain_rate_percent_per_min > 0:
        # More power needed (power_scale > 1) AND less usable capacity
        # (capacity_factor < 1) both make the battery drain FASTER.
        projected_drain_rate = baseline_drain_rate_percent_per_min * power_scale / capacity_factor

    return {
        "air_density_kg_m3": round(air_density(altitude_m, temperature_c), 4),
        "air_density_vs_sea_level_percent": round(
            (air_density(altitude_m, temperature_c) / SEA_LEVEL_DENSITY) * 100, 1
        ),
        "hover_power_increase_percent": round((power_scale - 1) * 100, 1),
        "battery_capacity_percent": round(capacity_factor * 100, 1),
        "baseline_drain_rate_percent_per_min": (
            round(baseline_drain_rate_percent_per_min, 3) if baseline_drain_rate_percent_per_min else None
        ),
        "projected_drain_rate_percent_per_min": (
            round(projected_drain_rate, 3) if projected_drain_rate else None
        ),
    }
