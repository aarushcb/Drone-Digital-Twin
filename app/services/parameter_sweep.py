"""
WHY THIS EXISTS:
A student learning multirotor design wants to ask "what if I used a
higher-KV motor, or a bigger prop, or added a heavier gimbal" and see the
real consequence -- not just an abstract formula. This reuses the exact
same physics already verified elsewhere in this app (required RPM/thrust
from motor_performance.py's dimensional propeller thrust formula, ISA air
density from environment_simulator.py) and evaluates it for both the
drone's CURRENTLY STORED spec and a SWEPT spec (the stored spec plus
requested deltas), so the comparison is apples-to-apples using one
physics model, not two.

WHAT'S NEW HERE (not already computed elsewhere):
1. HOVER EFFICIENCY (W/kg) -- total hover power divided by total mass.
   This is the standard "specific power" / power-loading metric used
   throughout multirotor UAV design literature to compare configurations
   independent of absolute size (a bigger drone needs more absolute power
   just from being bigger; W/kg normalizes that out) -- lower is more
   efficient. Not a new formula, just total_hover_power_watts / mass_kg,
   both already-verified quantities.

2. MAX ALTITUDE -- the altitude (under standard ISA atmosphere) at which
   this motor/prop/battery combination can no longer produce enough RPM
   to hover at all, i.e. where required_rpm(altitude) == max_available_rpm.
   Air density decreases monotonically with altitude (ISA barometric
   formula, already verified elsewhere in this app against real
   1.225kg/m^3 sea-level and ~0.909kg/m^3 @3000m reference values), and
   required_rpm scales as 1/sqrt(density) (motor_performance.py's
   dimensional thrust formula, T = Ct*rho*n^2*D^4, solved for n) --  so
   required_rpm increases monotonically with altitude, making this a
   textbook root-finding problem (exactly one altitude solves
   required_rpm(h) = max_available_rpm) solved here by BISECTION, a
   standard, guaranteed-to-converge numerical method for a monotonic
   function with a known bracketing interval -- not a new physics model,
   just applying a standard numerical technique to already-verified
   physics. "Standard ISA atmosphere" specifically means the density at
   altitude h is evaluated at the STANDARD lapse-rate temperature
   (15C - 0.0065*h, the same formula test_environment_simulator.py
   already uses to verify air_density's 3000m reference value) rather
   than an arbitrary fixed temperature -- an honest, documented choice,
   not a claim about a specific day's real weather.

WHAT WAS VERIFIED BEFORE SHIPPING:
- max_altitude_m's bisection result was cross-checked by directly calling
  required_rpm_for_thrust() at the returned altitude and confirming it
  lands within 1 RPM of max_available_rpm (i.e. the root-find actually
  converged to the right answer, not just "ran without crashing") --
  see test_parameter_sweep.py.
- hover_efficiency_w_per_kg was spot-checked by hand for the same
  reference scenario already verified in test_motor_performance.py
  (1.2kg quad, 10in props, 920KV, 4S, sea level -> 71.6W total hover
  power / 1.2kg = ~59.7 W/kg).
"""

import math

from app.services.motor_performance import (
    required_rpm_for_thrust,
    ideal_hover_power_watts,
    PROP_EFFICIENCY,
    LIPO_CELL_NOMINAL_VOLTAGE,
)
from app.services.environment_simulator import air_density as isa_air_density

MAX_ALTITUDE_SEARCH_CEILING_M = 9000  # matches EnvironmentSimulationRequest's own troposphere validity bound
_STANDARD_LAPSE_RATE_C_PER_M = 0.0065  # same standard ISA lapse rate already used/verified in environment_simulator.py


def _standard_temp_at_altitude_c(altitude_m: float) -> float:
    """Standard ISA atmosphere temperature at altitude -- same formula
    already used to verify air_density's 3000m reference value in
    test_environment_simulator.py."""
    return 15.0 - _STANDARD_LAPSE_RATE_C_PER_M * altitude_m


def _find_max_altitude_m(thrust_per_motor_n: float, diameter_m: float, max_available_rpm: float) -> dict:
    """
    Bisects for the altitude at which required_rpm(altitude) crosses
    max_available_rpm, under standard ISA atmosphere. Returns a dict
    flagging the two honest edge cases (infeasible even at sea level; or
    still feasible at the search ceiling, so the true ceiling -- if a
    finite one even exists within the troposphere -- is at least that
    high but not pinned down exactly) rather than returning a fabricated
    precise number in either case.
    """
    def required_rpm_at(altitude_m: float) -> float:
        density = isa_air_density(altitude_m, _standard_temp_at_altitude_c(altitude_m))
        return required_rpm_for_thrust(thrust_per_motor_n, density, diameter_m)

    rpm_at_sea_level = required_rpm_at(0)
    if rpm_at_sea_level > max_available_rpm:
        return {"max_altitude_m": 0.0, "feasible_at_sea_level": False, "search_capped": False}

    rpm_at_ceiling = required_rpm_at(MAX_ALTITUDE_SEARCH_CEILING_M)
    if rpm_at_ceiling <= max_available_rpm:
        return {
            "max_altitude_m": float(MAX_ALTITUDE_SEARCH_CEILING_M),
            "feasible_at_sea_level": True,
            "search_capped": True,  # still feasible at the search ceiling -- true max altitude (if any, within the troposphere) is at least this
        }

    lo, hi = 0.0, float(MAX_ALTITUDE_SEARCH_CEILING_M)
    for _ in range(40):  # more than enough iterations for sub-millimeter convergence on a ~9000m bracket
        mid = (lo + hi) / 2
        if required_rpm_at(mid) <= max_available_rpm:
            lo = mid
        else:
            hi = mid
    return {"max_altitude_m": round(lo, 1), "feasible_at_sea_level": True, "search_capped": False}


def analyze_sweep_point(
    mass_kg: float,
    motor_count: int,
    propeller_diameter_in: float,
    motor_kv: float,
    battery_cells: int,
    battery_capacity_mah: float,
    air_density: float = None,
) -> dict:
    """
    Full analysis for one point in spec-space -- reuses motor_performance.py's
    dimensional thrust/power formulas exactly as analyze_motor_performance()
    does, then adds hover_efficiency_w_per_kg and max_altitude_m on top.
    air_density defaults to sea level (1.225 kg/m^3) if not given, since the
    parameter sweep is about comparing SPECS, not simulating a specific
    environment (that's environment_simulator.py's job).
    """
    if air_density is None:
        air_density = 1.225

    diameter_m = propeller_diameter_in * 0.0254
    gravity = 9.81
    thrust_per_motor_n = (mass_kg * gravity) / motor_count

    required_rpm = required_rpm_for_thrust(thrust_per_motor_n, air_density, diameter_m)

    nominal_voltage = battery_cells * LIPO_CELL_NOMINAL_VOLTAGE
    max_available_rpm = motor_kv * nominal_voltage
    feasible = required_rpm <= max_available_rpm

    ideal_power_per_motor = ideal_hover_power_watts(thrust_per_motor_n, air_density, diameter_m)
    actual_power_per_motor = ideal_power_per_motor / PROP_EFFICIENCY
    total_power_watts = actual_power_per_motor * motor_count

    hover_efficiency_w_per_kg = total_power_watts / mass_kg if mass_kg > 0 else None

    energy_available_wh = (battery_capacity_mah / 1000) * nominal_voltage
    estimated_flight_time_minutes = (
        (energy_available_wh / total_power_watts) * 60 if total_power_watts > 0 else None
    )

    altitude_result = _find_max_altitude_m(thrust_per_motor_n, diameter_m, max_available_rpm)

    return {
        "required_rpm": round(required_rpm),
        "max_available_rpm": round(max_available_rpm),
        "feasible": feasible,
        "total_hover_power_watts": round(total_power_watts, 1),
        "hover_efficiency_w_per_kg": round(hover_efficiency_w_per_kg, 2) if hover_efficiency_w_per_kg else None,
        "estimated_flight_time_minutes": (
            round(estimated_flight_time_minutes, 1) if estimated_flight_time_minutes else None
        ),
        "max_altitude_m": altitude_result["max_altitude_m"],
        "altitude_search_capped": altitude_result["search_capped"],
    }


def analyze_parameter_sweep(
    mass_kg: float,
    motor_count: int,
    propeller_diameter_in: float,
    motor_kv: float,
    battery_cells: int,
    battery_capacity_mah: float,
    mass_delta_kg: float = 0.0,
    propeller_diameter_delta_in: float = 0.0,
    motor_kv_delta: float = 0.0,
    battery_cells_delta: int = 0,
) -> dict:
    """
    Computes the same analysis for the drone's CURRENT stored spec and a
    SWEPT spec (current + the requested deltas), so the Flutter UI can
    show a direct current-vs-swept comparison. battery_capacity_mah is
    intentionally not sweepable here -- capacity is a purchasing decision,
    not one of the four design parameters this feature is scoped to
    (motor KV, propeller diameter, battery cell count, total mass).
    """
    current = analyze_sweep_point(
        mass_kg=mass_kg,
        motor_count=motor_count,
        propeller_diameter_in=propeller_diameter_in,
        motor_kv=motor_kv,
        battery_cells=battery_cells,
        battery_capacity_mah=battery_capacity_mah,
    )

    swept_mass = max(0.01, mass_kg + mass_delta_kg)
    swept_diameter = max(0.1, propeller_diameter_in + propeller_diameter_delta_in)
    swept_kv = max(1.0, motor_kv + motor_kv_delta)
    swept_cells = max(1, round(battery_cells + battery_cells_delta))

    swept = analyze_sweep_point(
        mass_kg=swept_mass,
        motor_count=motor_count,
        propeller_diameter_in=swept_diameter,
        motor_kv=swept_kv,
        battery_cells=swept_cells,
        battery_capacity_mah=battery_capacity_mah,
    )

    return {
        "current": current,
        "current_spec": {
            "mass_kg": mass_kg,
            "propeller_diameter_in": propeller_diameter_in,
            "motor_kv": motor_kv,
            "battery_cells": battery_cells,
        },
        "swept": swept,
        "swept_spec": {
            "mass_kg": round(swept_mass, 3),
            "propeller_diameter_in": round(swept_diameter, 2),
            "motor_kv": round(swept_kv, 1),
            "battery_cells": swept_cells,
        },
    }
