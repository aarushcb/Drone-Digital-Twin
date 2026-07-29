"""
WHY THIS FILE EXISTS:
Upgrades the Environment Simulator from reporting only a RELATIVE power
change (e.g. "+20% more power needed") to computing REAL, absolute values:
actual required RPM, actual watts drawn, actual amps, and an actual
estimated flight time in minutes -- when the drone's motor/propeller specs
are on file.

WHERE EACH CONSTANT COMES FROM (same standard as the rest of this app --
citing exactly what's real vs. what's a documented published approximation):

1. STATIC THRUST COEFFICIENT (Ct) -- the standard dimensional propeller
   thrust formula, T = Ct * rho * n^2 * D^4 (n in rev/sec, D in meters),
   is the real formula used by published drone thrust calculators and the
   UIUC propeller test database. Ct = 0.105 used here is the MIDPOINT of
   the published static (hover, zero-airspeed) range for typical
   multirotor propellers (cited range: Ct ~0.09-0.12 from aggregated UIUC
   test data, as referenced in published drone performance calculators).
   This is a representative published value, not a per-unique-propeller
   individually measured curve -- an honest middle ground between pure
   idealized theory and a full manufacturer datasheet lookup we don't
   have access to.

2. PROPELLER EFFICIENCY -- 0.80, the commonly cited "good starting value"
   for quality plastic multirotor propellers in published thrust/power
   calculators (carbon fiber props cited up to 0.85, micro-drone props
   as low as 0.70-0.75).

3. LiPo NOMINAL CELL VOLTAGE -- 3.7V, the universal standard nominal
   voltage rating for a single LiPo cell (used industry-wide for "S"
   count ratings, e.g. 4S = 14.8V nominal).

VERIFIED before shipping: the required-RPM formula was checked against a
realistic reference scenario (1.2kg quad, 10" props) and produced ~4449
RPM per motor -- squarely within the real-world range small multirotor
drones actually hover at.
"""

import math

CT_STATIC = 0.105          # published UIUC-derived static thrust coefficient (midpoint of cited 0.09-0.12 range)
PROP_EFFICIENCY = 0.80     # published typical value for quality multirotor propellers
LIPO_CELL_NOMINAL_VOLTAGE = 3.7  # standard nominal LiPo cell voltage


def required_rpm_for_thrust(thrust_per_motor_n: float, air_density: float, diameter_m: float) -> float:
    """Solves T = Ct * rho * n^2 * D^4 for n (rev/sec), returns RPM."""
    n_rev_per_sec = math.sqrt(thrust_per_motor_n / (CT_STATIC * air_density * diameter_m**4))
    return n_rev_per_sec * 60


def ideal_hover_power_watts(thrust_per_motor_n: float, air_density: float, diameter_m: float) -> float:
    """
    Actuator disk (momentum theory) ideal power: P = T^1.5 / sqrt(2 * rho * A).
    Same formula already verified in app/services/environment_simulator.py
    for the relative-power-change calculation -- reused here for an
    absolute value instead of a ratio.
    """
    area_m2 = math.pi * (diameter_m / 2) ** 2
    return (thrust_per_motor_n ** 1.5) / math.sqrt(2 * air_density * area_m2)


def analyze_motor_performance(
    mass_kg: float,
    motor_count: int,
    propeller_diameter_in: float,
    motor_kv: float,
    battery_cells: int,
    battery_capacity_mah: float,
    air_density: float,
) -> dict:
    """
    Full real-physics analysis: given complete motor/propeller/battery
    specs, computes actual required RPM, whether the motor can actually
    reach it, real power/current draw, and a real estimated flight time.
    """
    diameter_m = propeller_diameter_in * 0.0254
    gravity = 9.81
    total_hover_thrust_n = mass_kg * gravity
    thrust_per_motor_n = total_hover_thrust_n / motor_count

    required_rpm = required_rpm_for_thrust(thrust_per_motor_n, air_density, diameter_m)

    nominal_voltage = battery_cells * LIPO_CELL_NOMINAL_VOLTAGE
    max_available_rpm = motor_kv * nominal_voltage

    # WHY THIS FEASIBILITY CHECK IS GENUINELY NEW, NOT SOMETHING THE OLD
    # RELATIVE-RATIO MODEL COULD EVER TELL YOU: the old model always
    # implicitly assumed the drone COULD hover, and only asked "how much
    # MORE power would that take." This asks the more fundamental
    # question first -- can the selected motor/prop/battery combination
    # physically produce enough RPM to hover at all under these
    # conditions.
    feasible = required_rpm <= max_available_rpm
    rpm_margin_percent = ((max_available_rpm - required_rpm) / max_available_rpm) * 100

    ideal_power_per_motor = ideal_hover_power_watts(thrust_per_motor_n, air_density, diameter_m)
    actual_power_per_motor = ideal_power_per_motor / PROP_EFFICIENCY
    total_power_watts = actual_power_per_motor * motor_count

    hover_current_amps = total_power_watts / nominal_voltage if nominal_voltage > 0 else None

    energy_available_wh = (battery_capacity_mah / 1000) * nominal_voltage
    estimated_flight_time_minutes = (
        (energy_available_wh / total_power_watts) * 60 if total_power_watts > 0 else None
    )

    return {
        "required_rpm": round(required_rpm),
        "max_available_rpm": round(max_available_rpm),
        "feasible": feasible,
        "rpm_margin_percent": round(rpm_margin_percent, 1),
        "nominal_voltage": round(nominal_voltage, 1),
        "total_hover_power_watts": round(total_power_watts, 1),
        "hover_current_amps": round(hover_current_amps, 2) if hover_current_amps else None,
        "estimated_flight_time_minutes": (
            round(estimated_flight_time_minutes, 1) if estimated_flight_time_minutes else None
        ),
    }


def has_complete_motor_specs(drone) -> bool:
    """Checks whether a drone has all the fields needed for this analysis."""
    return all([
        drone.mass_kg,
        drone.motor_count,
        drone.propeller_diameter_in,
        drone.motor_kv,
        drone.battery_cells,
        drone.battery_capacity_mah,
    ])
