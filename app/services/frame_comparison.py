"""
WHY THIS EXISTS:
A student choosing between a quad, hex, octo, or fixed-wing airframe for
the SAME motor/propeller/battery choice wants to see the real tradeoffs
side by side -- more motors means more redundancy and thrust margin but
also more weight and drag; a fixed-wing needs none of the hover thrust
margin a rotorcraft does, but pays for that with completely different
flight mechanics. This reuses parameter_sweep.py's exact hover-physics
formulas (mass/motor-count/prop/battery -> required RPM, hover power,
efficiency, flight time, max altitude), varying only motor_count per
frame type, and adds two genuinely new things: thrust-to-weight ratio
(max available thrust, not the trivial hover-equals-weight ratio) and
turn rate, derived from it via a real, standard aviation formula.

WHERE MOTOR COUNT PER FRAME COMES FROM:
quad=4, hex=6, octo=8 (standard multirotor naming), fixed_wing=1 (a
single tractor/pusher motor -- the common configuration for small fixed-
wing UAVs; twin-engine fixed-wing exists but is the less common case,
documented as this module's assumption, not a universal truth).

THRUST-TO-WEIGHT RATIO -- MAX AVAILABLE THRUST, NOT HOVER THRUST:
At hover, thrust trivially equals weight by definition (T/W=1) --
useless for comparison. The metric that actually matters is the ratio
at MAXIMUM available RPM (motor_kv * pack voltage) to weight -- the real,
standard way "thrust-to-weight ratio" is used throughout the hobby/
industry (e.g. "a 2:1 thrust-to-weight racing quad"), using
motor_performance.py's own T=Ct*rho*n^2*D^4 formula evaluated at max
RPM instead of required RPM.

MAX TILT ANGLE AND TURN RATE -- REAL, STANDARD FORMULAS:
For a ROTORCRAFT (quad/hex/octo), the maximum sustainable tilt angle is
set by how much of the available thrust is left over once enough is
reserved (as the VERTICAL component of tilted thrust) to still support
weight: at tilt angle theta, T*cos(theta) must still equal W, so the
tilt angle that uses ALL available thrust is:
    max_tilt_deg = arccos(1 / thrust_to_weight_ratio)
This is exact trigonometry, not a fitted approximation -- and matches a
widely-cited reference figure in multirotor/FPV literature: a 2:1
thrust-to-weight ratio permits up to 60 degrees of tilt while still
generating enough vertical thrust to hover (arccos(1/2) = 60 degrees
exactly) -- used here as a hand-verifiable sanity check, not an external
citation dependency.

For FIXED-WING, this thrust-margin relationship doesn't apply -- a
fixed-wing aircraft's turn/bank limit is set by available LIFT (wing
aerodynamics/structural load factor), not motor thrust margin, and this
app has no wing area/aspect ratio data to compute that properly (adding
those fields would be a schema change outside this feature's scope).
Instead, a representative "typical maneuvering bank angle" of 25 degrees
is assumed for fixed_wing -- documented honestly as a representative
assumption, not a thrust-derived result, the same "typical value,
clearly flagged" standard as this app's other representative constants.

Once a max tilt/bank angle is established (by whichever method fits the
airframe), the ACTUAL TURN RATE uses the same real formula for every
frame type -- the standard COORDINATED TURN RATE relationship from flight
dynamics (see e.g. the FAA's own Airplane Flying Handbook's "rate of
turn" treatment, or any flight dynamics/multirotor controls text):
    turn_rate_deg_s = degrees(g * tan(bank_angle) / velocity)
(the requested rough estimate "(max_tilt_angle * gravity) / velocity" is
the SMALL-ANGLE approximation of this exact formula, tan(theta) ~ theta
in radians for small theta -- the exact tan() form is used here instead
since it's equally simple to compute and more accurate at the larger
tilt angles a high-thrust-margin racing-style quad can actually reach).

PRACTICAL TILT CAP -- WHY THE THRUST-DERIVED ANGLE ISN'T USED UNCAPPED:
As thrust-to-weight ratio gets very large, arccos(1/TWR) approaches 90
degrees, and tan(theta) in the turn-rate formula diverges toward
infinity right along with it -- a mathematically correct consequence of
the formula, but not a realistic flight maneuver (no real vehicle
sustains a near-90-degree bank in level flight, and real flight
controller firmware always enforces a configured practical tilt limit
well below the absolute thrust ceiling regardless of how much thrust
margin is technically available -- e.g. PX4's MPC_TILTMAX_AIR and
ArduPilot's ANGLE_MAX parameters, both real, commonly documented in the
45-80 degree range for actual multirotor configurations). This module
caps the thrust-derived tilt angle at PRACTICAL_MAX_TILT_DEG=60 degrees
(a representative value within that real documented firmware range) --
so a very high-thrust-margin build correctly shows a large max tilt
angle up to that realistic practical ceiling, rather than an unbounded
number that blows up toward infinity in the turn-rate formula.

HONEST LIMITATION -- HOVER EFFICIENCY/ENDURANCE/MAX ALTITUDE FOR
FIXED-WING: these are computed using the exact same hover-physics
formulas as the rotorcraft frames (reusing parameter_sweep.py directly),
which is NOT how a fixed-wing aircraft is actually flown (it doesn't
hover -- lift comes from forward airspeed over a wing, and real cruise
endurance depends on lift-to-drag ratio and wing loading, data this
app's Drone model doesn't store). These numbers are reported for
fixed_wing anyway, clearly flagged via is_rotorcraft=False, specifically
because they're still genuinely informative for comparison -- they show
how thrust-inefficient a cruise-optimized propulsion setup would be if
forced to hover, which is exactly why fixed-wing designs don't try to.
They are NOT a claim about this aircraft's real cruise endurance.

WHAT WAS VERIFIED BEFORE SHIPPING:
- max_tilt_deg was hand-verified against the widely-cited "2:1
  thrust-to-weight permits 60 degree tilt" reference figure:
  arccos(1/2) = 60.0 degrees exactly.
- turn_rate_deg_s was cross-checked against the small-angle approximation
  the task itself suggested (theta*g/V) at a small tilt angle, confirming
  the two agree closely there and diverge (as expected) only at larger
  angles where the small-angle approximation itself breaks down.
"""

import math
from typing import List, Optional

from app.services.parameter_sweep import analyze_sweep_point
from app.services.motor_performance import CT_STATIC, LIPO_CELL_NOMINAL_VOLTAGE

FRAME_MOTOR_COUNTS = {"quad": 4, "hex": 6, "octo": 8, "fixed_wing": 1}
ASSUMED_FIXED_WING_BANK_ANGLE_DEG = 25.0
PRACTICAL_MAX_TILT_DEG = 60.0  # representative real firmware-configured practical tilt limit (PX4/ArduPilot typical 45-80 deg range)
DEFAULT_CRUISE_VELOCITY_MPS = {"quad": 8.0, "hex": 8.0, "octo": 8.0, "fixed_wing": 15.0}
GRAVITY_MPS2 = 9.81

VALID_FRAME_TYPES = set(FRAME_MOTOR_COUNTS.keys())


def _max_available_thrust_n(
    motor_count: int, propeller_diameter_in: float, motor_kv: float, battery_cells: int, air_density: float
) -> float:
    """Total thrust at maximum available RPM (motor_kv * pack voltage) --
    the same static thrust formula motor_performance.py uses for required
    thrust, evaluated at max RPM instead, giving the real "how much
    thrust could this setup produce at full throttle" figure."""
    diameter_m = propeller_diameter_in * 0.0254
    nominal_voltage = battery_cells * LIPO_CELL_NOMINAL_VOLTAGE
    max_rpm = motor_kv * nominal_voltage
    n_rev_per_sec = max_rpm / 60
    thrust_per_motor = CT_STATIC * air_density * (n_rev_per_sec ** 2) * (diameter_m ** 4)
    return thrust_per_motor * motor_count


def compare_frames(
    frame_types: List[str],
    mass_kg: float,
    propeller_diameter_in: float,
    motor_kv: float,
    battery_cells: int,
    battery_capacity_mah: float,
    air_density: float = 1.225,
    velocity_mps: Optional[float] = None,
) -> dict:
    """
    Compares the requested frame types for the SAME motor/propeller/
    battery/mass choice, varying only motor_count (and, for fixed_wing,
    the physically appropriate max-bank-angle method) per frame.
    """
    results = {}
    for frame in frame_types:
        motor_count = FRAME_MOTOR_COUNTS[frame]
        is_rotorcraft = frame != "fixed_wing"

        hover_analysis = analyze_sweep_point(
            mass_kg=mass_kg,
            motor_count=motor_count,
            propeller_diameter_in=propeller_diameter_in,
            motor_kv=motor_kv,
            battery_cells=battery_cells,
            battery_capacity_mah=battery_capacity_mah,
            air_density=air_density,
        )

        max_thrust_n = _max_available_thrust_n(
            motor_count, propeller_diameter_in, motor_kv, battery_cells, air_density
        )
        weight_n = mass_kg * GRAVITY_MPS2
        thrust_to_weight_ratio = max_thrust_n / weight_n if weight_n > 0 else 0.0

        if is_rotorcraft:
            thrust_derived_tilt_deg = (
                math.degrees(math.acos(min(1.0, 1 / thrust_to_weight_ratio)))
                if thrust_to_weight_ratio >= 1
                else 0.0
            )
            max_tilt_deg = min(thrust_derived_tilt_deg, PRACTICAL_MAX_TILT_DEG)
        else:
            max_tilt_deg = ASSUMED_FIXED_WING_BANK_ANGLE_DEG

        velocity = velocity_mps if velocity_mps else DEFAULT_CRUISE_VELOCITY_MPS[frame]
        turn_rate_deg_s = math.degrees(GRAVITY_MPS2 * math.tan(math.radians(max_tilt_deg)) / velocity)

        results[frame] = {
            "motor_count": motor_count,
            "is_rotorcraft": is_rotorcraft,
            "thrust_to_weight_ratio": round(thrust_to_weight_ratio, 2),
            "max_tilt_angle_deg": round(max_tilt_deg, 1),
            "turn_rate_deg_s": round(turn_rate_deg_s, 1),
            "hover_efficiency_w_per_kg": hover_analysis["hover_efficiency_w_per_kg"],
            "estimated_flight_time_minutes": hover_analysis["estimated_flight_time_minutes"],
            "max_altitude_m": hover_analysis["max_altitude_m"],
            "altitude_search_capped": hover_analysis["altitude_search_capped"],
            "feasible_hover": hover_analysis["feasible"],
        }

    return {
        "shared_spec": {
            "mass_kg": mass_kg,
            "propeller_diameter_in": propeller_diameter_in,
            "motor_kv": motor_kv,
            "battery_cells": battery_cells,
            "battery_capacity_mah": battery_capacity_mah,
        },
        "frames": results,
    }
