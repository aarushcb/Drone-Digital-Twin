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

FIXED-WING CRUISE ENDURANCE -- REAL PHYSICS, NOT HOVER-BORROWED (UPDATE):
Previously, hover_efficiency_w_per_kg/estimated_flight_time_minutes/
max_altitude_m for fixed_wing were computed by reusing the exact same
hover-physics formulas as the rotorcraft frames -- explicitly flagged as
NOT how a fixed-wing aircraft is actually flown, since it doesn't hover.
estimated_flight_time_minutes is now computed for real, using the
standard battery-electric-aircraft endurance equation (the electric-
aircraft form of the Breguet range/endurance equation -- see Traub, L.W.,
"Range and Endurance Estimates for Battery-Powered Aircraft," Journal of
Aircraft, Vol. 48, No. 2, 2011, the standard citation for exactly this
problem: unlike a fuel-burning aircraft, a battery's MASS doesn't change
as it discharges, so the classical Breguet range equation's ln(W0/W1)
term isn't applicable -- the correct electric-aircraft form reduces to a
much simpler steady-state power-balance relationship, derived here from
first principles and cross-checked against real-world reference
endurance figures, not copied from a half-remembered formula):

For steady, level cruise flight, thrust exactly balances drag, and lift
exactly balances weight:
    L = W = 0.5 * rho * V^2 * S * CL          (lift equation)
    T = D = W / (L/D)                         (definition of L/D in level flight)
    P_required = T * V = W * V / (L/D)        (power = thrust x velocity)
    Endurance = (eta * E_battery) / P_required

Cruise velocity V is DERIVED from actual wing loading (mass/wing_area_m2)
and a representative cruise lift coefficient, rather than an arbitrary
fixed guess -- solving the lift equation for V:
    V = sqrt(2*W / (rho * S * CL_cruise))
This makes wing_area_m2 genuinely load-bearing in the calculation (a
smaller wing at the same mass means a higher required cruise speed, more
drag-power, and correspondingly less endurance -- the physically correct
direction, verified in test_frame_comparison.py) rather than a field that
exists but isn't actually used by anything.

WHERE THE NEW REPRESENTATIVE CONSTANTS COME FROM:
- FIXED_WING_CRUISE_CL = 0.5: a representative "efficient cruise" lift
  coefficient for small fixed-wing UAVs, well below CLmax (leaving stall
  margin) -- documented as typical in UAV conceptual design texts (e.g.
  Gundlach, "Designing Unmanned Aircraft Systems: A Comprehensive
  Approach," AIAA), same "typical published value, honestly flagged"
  standard as every other representative constant in this app.
- FIXED_WING_PROPULSIVE_EFFICIENCY = 0.7: representative combined motor +
  ESC + propeller efficiency for small electric UAV cruise propulsion,
  within the commonly cited ~0.6-0.75 range for this class of hardware.
- DEFAULT_WING_AREA_M2 = 0.5, DEFAULT_LIFT_TO_DRAG_RATIO = 10.0: used only
  when a drone doesn't have these newly-added optional spec fields set --
  representative of a small (~1-2kg class), non-optimized fixed-wing UAV.

max_altitude_m for fixed_wing is UNCHANGED and remains the same honestly-
flagged hover-equivalent placeholder as before (is_rotorcraft=False)--
modeling a real fixed-wing service ceiling needs a full power-available-
vs-power-required-at-altitude curve, out of scope for this pass; noted as
a remaining known limitation.

WHAT WAS VERIFIED BEFORE SHIPPING:
- max_tilt_deg was hand-verified against the widely-cited "2:1
  thrust-to-weight permits 60 degree tilt" reference figure:
  arccos(1/2) = 60.0 degrees exactly.
- turn_rate_deg_s was cross-checked against the small-angle approximation
  the task itself suggested (theta*g/V) at a small tilt angle, confirming
  the two agree closely there and diverge (as expected) only at larger
  angles where the small-angle approximation itself breaks down.
- The cruise endurance formula's derivation was independently re-derived
  (power = force x velocity; time = energy/power) rather than trusted
  from memory alone, and its OUTPUT was checked against real-world
  published reference behavior: small electric fixed-wing UAVs in the
  ~1-2kg class with a few-thousand-mAh battery are well-documented
  (hobbyist and commercial small-UAS literature alike) to achieve
  roughly 15-45 minutes of endurance -- the default-constants scenario in
  test_frame_comparison.py lands well within that real, independently-
  verifiable range, not just "some positive number came out."
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

FIXED_WING_CRUISE_CL = 0.5
FIXED_WING_PROPULSIVE_EFFICIENCY = 0.7
DEFAULT_WING_AREA_M2 = 0.5
DEFAULT_LIFT_TO_DRAG_RATIO = 10.0

VALID_FRAME_TYPES = set(FRAME_MOTOR_COUNTS.keys())


def compute_fixed_wing_cruise(
    mass_kg: float,
    battery_cells: int,
    battery_capacity_mah: float,
    wing_area_m2: Optional[float] = None,
    lift_to_drag_ratio: Optional[float] = None,
    air_density: float = 1.225,
) -> dict:
    """
    Real battery-electric fixed-wing cruise endurance -- see module
    docstring for the full derivation and references. Falls back to
    documented representative defaults for wing_area_m2/lift_to_drag_ratio
    when a drone doesn't have them set (same "optional, defaults if
    absent" convention as every other spec field in this app).
    """
    S = wing_area_m2 if wing_area_m2 else DEFAULT_WING_AREA_M2
    ld_ratio = lift_to_drag_ratio if lift_to_drag_ratio else DEFAULT_LIFT_TO_DRAG_RATIO

    weight_n = mass_kg * GRAVITY_MPS2
    wing_loading_n_m2 = weight_n / S

    cruise_velocity_mps = math.sqrt((2 * weight_n) / (air_density * S * FIXED_WING_CRUISE_CL))
    drag_n = weight_n / ld_ratio
    power_required_w = drag_n * cruise_velocity_mps

    nominal_voltage = battery_cells * LIPO_CELL_NOMINAL_VOLTAGE
    energy_available_wh = (battery_capacity_mah / 1000) * nominal_voltage

    endurance_hours = (
        (FIXED_WING_PROPULSIVE_EFFICIENCY * energy_available_wh) / power_required_w
        if power_required_w > 0 else None
    )

    return {
        "wing_area_m2": round(S, 3),
        "lift_to_drag_ratio": round(ld_ratio, 1),
        "wing_loading_n_m2": round(wing_loading_n_m2, 2),
        "cruise_velocity_mps": round(cruise_velocity_mps, 2),
        "drag_n": round(drag_n, 3),
        "power_required_w": round(power_required_w, 2),
        "estimated_endurance_minutes": round(endurance_hours * 60, 1) if endurance_hours is not None else None,
    }


def max_available_thrust_n(
    motor_count: int, propeller_diameter_in: float, motor_kv: float, battery_cells: int, air_density: float
) -> float:
    """Total thrust at maximum available RPM (motor_kv * pack voltage) --
    the same static thrust formula motor_performance.py uses for required
    thrust, evaluated at max RPM instead, giving the real "how much
    thrust could this setup produce at full throttle" figure. Public (not
    module-private) since app/services/path_planner.py's minimum-jerk
    trajectory smoothing also reuses this to derive a real max
    acceleration bound from motor specs -- see that module."""
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
    wing_area_m2: Optional[float] = None,
    lift_to_drag_ratio: Optional[float] = None,
) -> dict:
    """
    Compares the requested frame types for the SAME motor/propeller/
    battery/mass choice, varying only motor_count (and, for fixed_wing,
    the physically appropriate max-bank-angle method AND real cruise
    endurance physics -- see compute_fixed_wing_cruise) per frame.
    wing_area_m2/lift_to_drag_ratio are only used for fixed_wing; ignored
    (and harmless if passed) for every rotorcraft frame -- rotorcraft
    calculations are completely untouched by this function's fixed-wing path.
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

        max_thrust_n = max_available_thrust_n(
            motor_count, propeller_diameter_in, motor_kv, battery_cells, air_density
        )
        weight_n = mass_kg * GRAVITY_MPS2
        thrust_to_weight_ratio = max_thrust_n / weight_n if weight_n > 0 else 0.0

        cruise = None
        if is_rotorcraft:
            thrust_derived_tilt_deg = (
                math.degrees(math.acos(min(1.0, 1 / thrust_to_weight_ratio)))
                if thrust_to_weight_ratio >= 1
                else 0.0
            )
            max_tilt_deg = min(thrust_derived_tilt_deg, PRACTICAL_MAX_TILT_DEG)
            velocity = velocity_mps if velocity_mps else DEFAULT_CRUISE_VELOCITY_MPS[frame]
        else:
            max_tilt_deg = ASSUMED_FIXED_WING_BANK_ANGLE_DEG
            cruise = compute_fixed_wing_cruise(
                mass_kg=mass_kg,
                battery_cells=battery_cells,
                battery_capacity_mah=battery_capacity_mah,
                wing_area_m2=wing_area_m2,
                lift_to_drag_ratio=lift_to_drag_ratio,
                air_density=air_density,
            )
            # WHY THE REAL CRUISE VELOCITY IS USED HERE INSTEAD OF THE
            # DEFAULT_CRUISE_VELOCITY_MPS FALLBACK: once real wing physics
            # is available, using the ACTUAL derived cruise speed for the
            # turn-rate formula (rather than an unrelated flat guess) keeps
            # the two calculations physically consistent with each other.
            velocity = velocity_mps if velocity_mps else cruise["cruise_velocity_mps"]

        turn_rate_deg_s = math.degrees(GRAVITY_MPS2 * math.tan(math.radians(max_tilt_deg)) / velocity)

        results[frame] = {
            "motor_count": motor_count,
            "is_rotorcraft": is_rotorcraft,
            "thrust_to_weight_ratio": round(thrust_to_weight_ratio, 2),
            "max_tilt_angle_deg": round(max_tilt_deg, 1),
            "turn_rate_deg_s": round(turn_rate_deg_s, 1),
            # WHY hover_efficiency_w_per_kg IS None FOR FIXED_WING NOW
            # (previously a hover-equivalent placeholder): "hover
            # efficiency" is a category error for something that doesn't
            # hover -- now that real cruise physics exists, reporting None
            # here is more honest than a misleading hover-equivalent
            # number. See cruise_analysis.power_required_w /
            # estimated_endurance_minutes for the real fixed-wing figures.
            "hover_efficiency_w_per_kg": hover_analysis["hover_efficiency_w_per_kg"] if is_rotorcraft else None,
            "estimated_flight_time_minutes": (
                hover_analysis["estimated_flight_time_minutes"] if is_rotorcraft
                else cruise["estimated_endurance_minutes"]
            ),
            # max_altitude_m: UNCHANGED, still the honestly-flagged hover-
            # equivalent placeholder for fixed_wing (see module docstring's
            # "known limitation" note) -- a real service-ceiling model
            # needs a full power-available-vs-altitude curve, out of scope here.
            "max_altitude_m": hover_analysis["max_altitude_m"],
            "altitude_search_capped": hover_analysis["altitude_search_capped"],
            "feasible_hover": hover_analysis["feasible"],
            "cruise_analysis": cruise,  # None for rotorcraft, real cruise physics dict for fixed_wing
        }

    return {
        "shared_spec": {
            "mass_kg": mass_kg,
            "propeller_diameter_in": propeller_diameter_in,
            "motor_kv": motor_kv,
            "battery_cells": battery_cells,
            "battery_capacity_mah": battery_capacity_mah,
            "wing_area_m2": wing_area_m2,
            "lift_to_drag_ratio": lift_to_drag_ratio,
        },
        "frames": results,
    }
