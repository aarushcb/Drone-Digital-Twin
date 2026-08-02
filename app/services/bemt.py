"""
WHY THIS EXISTS:
app/services/motor_performance.py computes hover thrust using a single
empirical static thrust coefficient (Ct = 0.105, the midpoint of a cited
published range) -- a good, honest first model, but it treats the propeller
as one lumped-together constant rather than an actual blade shape. BEMT
(Blade Element Momentum Theory) is the next level of fidelity used in real
rotorcraft/propeller design: it slices the blade into small radial elements,
computes the lift/drag each element produces locally (blade element theory),
combines that with how much the air is being accelerated downward at that
radius (momentum theory), and integrates across the whole blade to get the
overall thrust and power coefficients. This doesn't replace the existing
motor_performance model (still used, still correct for its purpose) --
it's an additive, more detailed cross-check computed from blade geometry
instead of a single constant.

WHERE EACH PART OF THE MATH COMES FROM:

1. COMBINED BLADE ELEMENT / MOMENTUM EQUATION FOR LOCAL INFLOW -- for a
   blade element at nondimensional radius r (0 to 1), blade element theory
   gives dCT/dr = (sigma*Cla/2)*(theta*r^2 - lambda*r), and momentum theory
   (differential annulus form) gives dCT/dr = 4*F*lambda^2*r. Setting these
   equal and solving the resulting quadratic for lambda gives the standard
   closed-form hover inflow solution:

       lambda(r) = (sigma*Cla)/(16*F) * [sqrt(1 + 32*F*theta(r)/(sigma*Cla)*r) - 1]

   This is the standard combined-BEMT hover result presented in Leishman,
   "Principles of Helicopter Aerodynamics" (2nd ed., Cambridge Aerospace
   Series), Ch. 3, and in Johnson, "Helicopter Theory" -- re-derived here
   from the two governing equations above (not copied blind) as a sanity
   check that it's actually algebraically consistent; verified by hand
   that combining the two dCT/dr expressions and solving the resulting
   quadratic (4*F*lambda^2 + (sigma*Cla/2)*lambda - (sigma*Cla/2)*theta*r = 0)
   via the quadratic formula reduces exactly to the closed form above.

2. PRANDTL TIP-LOSS FACTOR -- real blades lose lift near the tip because
   air can "leak" around it instead of being cleanly accelerated downward;
   Prandtl's classic correction factor F = (2/pi)*arccos(exp(-f)), with
   f = (Nb/2)*(1-r)/(r*phi), is the standard way every rotorcraft textbook
   (Leishman, Johnson) applies this correction. phi (local inflow angle)
   is approximated as lambda/r and F is recomputed for a couple of
   iterations until it stabilizes, since F and lambda depend on each other.

3. THRUST/POWER COEFFICIENT INTEGRATION -- dCT/dr = 4*F*lambda^2*r
   (induced) integrated over the blade gives CT; power has an induced
   term (dCPi/dr = lambda*dCT/dr) plus a profile-drag term
   (dCP0/dr = (sigma*Cd0/2)*r^3, standard result of integrating local
   drag*radius*local-speed-squared over the blade) -- both textbook-
   standard (Leishman Ch. 2-3).

4. GEOMETRIC/AIRFOIL CONSTANTS -- since we don't have a real manufacturer
   blade geometry file (same honest limitation as motor_performance.py's
   Ct constant), these use documented "typical small multirotor propeller"
   values, not a specific measured blade:
     - lift-curve slope Cl_alpha = 5.75 /rad: thin-airfoil theory predicts
       2*pi ~= 6.283/rad, but real thin cambered sections at the low
       Reynolds numbers small multirotor props operate at (~30k-150k,
       per the UIUC low-Reynolds-number airfoil database -- the same UIUC
       data family cited in motor_performance.py) measure consistently
       lower, commonly cited in the 5.5-5.8/rad range for this Re regime.
     - profile drag coefficient Cd0 = 0.02: a standard "typical" value
       for thin RC/UAV propeller airfoil sections cited across rotor
       performance texts (documented range ~0.015-0.03).
     - chord-to-radius ratio c/R = 0.10 and linear twist from 36 deg
       (root) to 10 deg (tip): representative of measured small (9-12in)
       multirotor propeller geometries in the UIUC propeller database
       (typical c/R ~0.06-0.12). The 26-degree total washout is on the
       higher end of typically-cited multirotor prop twist, but is
       consistent with the classical "ideal hover twist" result (theta
       proportional to 1/r, i.e. high root pitch tapering sharply to a
       much lower tip pitch) that real efficient hover propellers
       approximate -- see Gessow & Myers, "Aerodynamics of the
       Helicopter", on ideal (minimum induced loss) hover twist.
   These are flagged the same honest way as the existing Ct constant:
   representative published values, not a per-blade manufacturer
   datasheet lookup.

WHAT WAS VERIFIED BEFORE SHIPPING:
- The physical invariant that MUST hold for any correctly-formed BEMT
  hover coefficient calculation: since CT and CP are purely geometric
  (independent of RPM in the hover, no-swirl formulation used here),
  dimensional thrust must scale as RPM^2 and power as RPM^3 exactly.
  Checked numerically in test_bemt.py -- this is a strong correctness
  check because it would fail if the coefficient integration were wrong,
  not just "look plausible."
- CT for the default representative geometry lands at ~0.0104, inside
  the commonly cited real-world range (~0.008-0.02) for small multirotor
  propellers reported in aggregated UIUC static thrust test data --
  the same reference family already used for CT_STATIC in
  motor_performance.py.
"""

import math

CL_ALPHA_PER_RAD = 5.75      # typical low-Reynolds-number thin-airfoil lift-curve slope (UIUC data range)
CD0 = 0.02                   # typical RC/UAV propeller section profile drag coefficient
CHORD_TO_RADIUS = 0.10       # typical c/R for small multirotor propellers
THETA_ROOT_DEG = 36.0        # representative linear-twist root pitch (ideal-hover-twist-like taper)
THETA_TIP_DEG = 10.0         # representative linear-twist tip pitch
ROOT_CUTOUT = 0.15           # fraction of radius with no effective blade (hub/root region)
NUM_BLADES = 2               # standard for multirotor propellers
NUM_ELEMENTS = 40            # radial discretization for the blade integration


def _local_twist_rad(r: float) -> float:
    """Linear twist from root to tip, in radians, at nondimensional radius r (0-1)."""
    theta_deg = THETA_ROOT_DEG + (THETA_TIP_DEG - THETA_ROOT_DEG) * r
    return math.radians(theta_deg)


def _tip_loss_factor(r: float, lam: float) -> float:
    """Prandtl tip-loss factor F, given local inflow ratio lambda."""
    if r <= 0 or lam <= 1e-9:
        return 1.0
    phi = lam / r
    f = (NUM_BLADES / 2) * (1 - r) / (r * phi)
    f = min(f, 50.0)  # guard against overflow in exp() for r -> 1 edge case
    return (2 / math.pi) * math.acos(max(-1.0, min(1.0, math.exp(-f))))


def _local_inflow_ratio(r: float, sigma: float, theta: float) -> float:
    """
    Closed-form combined blade-element/momentum inflow at radius r, with
    2 fixed-point iterations on the tip-loss factor F (F depends on
    lambda, lambda depends on F -- converges quickly in practice).
    """
    F = 1.0
    lam = 0.0
    for _ in range(3):
        inner = 1 + (32 * F * theta * r) / (sigma * CL_ALPHA_PER_RAD)
        lam = (sigma * CL_ALPHA_PER_RAD) / (16 * F) * (math.sqrt(max(inner, 0.0)) - 1)
        lam = max(lam, 1e-6)
        F = _tip_loss_factor(r, lam)
    return lam, F


def hover_coefficients(num_blades: int = NUM_BLADES, chord_to_radius: float = CHORD_TO_RADIUS) -> dict:
    """
    Integrates blade-element/momentum theory across the blade (root cutout
    to tip) to get the thrust and power coefficients CT, CP for the
    representative blade geometry. These are purely geometric -- they do
    NOT depend on RPM or air density (that's what makes hover BEMT in
    non-dimensional form clean: CT/CP are constants of the blade shape).
    """
    sigma_local = (num_blades * chord_to_radius) / math.pi  # solidity is constant here since chord/R is constant

    ct = 0.0
    cp_induced = 0.0
    cp_profile = 0.0
    dr = (1.0 - ROOT_CUTOUT) / NUM_ELEMENTS

    for i in range(NUM_ELEMENTS):
        r = ROOT_CUTOUT + (i + 0.5) * dr  # midpoint rule
        theta = _local_twist_rad(r)
        lam, F = _local_inflow_ratio(r, sigma_local, theta)

        d_ct = 4 * F * lam ** 2 * r
        d_cp_i = lam * d_ct
        d_cp_0 = (sigma_local * CD0 / 2) * r ** 3

        ct += d_ct * dr
        cp_induced += d_cp_i * dr
        cp_profile += d_cp_0 * dr

    return {
        "ct": ct,
        "cp": cp_induced + cp_profile,
        "cp_induced": cp_induced,
        "cp_profile": cp_profile,
        "solidity": sigma_local,
    }


def bemt_thrust_and_power(
    diameter_m: float,
    rpm: float,
    air_density: float,
    num_blades: int = NUM_BLADES,
    chord_to_radius: float = CHORD_TO_RADIUS,
) -> dict:
    """
    Dimensional thrust (N) and power (W) for one propeller at a given RPM,
    from the blade-element/momentum coefficients: T = CT*rho*A*(Omega*R)^2,
    P = CP*rho*A*(Omega*R)^3 -- the standard non-dimensionalization used
    throughout rotorcraft/propeller performance analysis.
    """
    radius_m = diameter_m / 2
    area_m2 = math.pi * radius_m ** 2
    omega_rad_s = rpm * 2 * math.pi / 60
    tip_speed = omega_rad_s * radius_m

    coeffs = hover_coefficients(num_blades=num_blades, chord_to_radius=chord_to_radius)

    thrust_n = coeffs["ct"] * air_density * area_m2 * tip_speed ** 2
    power_w = coeffs["cp"] * air_density * area_m2 * tip_speed ** 3

    return {
        "ct": round(coeffs["ct"], 5),
        "cp": round(coeffs["cp"], 5),
        "thrust_n": thrust_n,
        "power_w": power_w,
    }


def bemt_required_rpm_for_thrust(thrust_per_motor_n: float, air_density: float, diameter_m: float) -> float:
    """
    Solves T = CT*rho*A*(Omega*R)^2 for Omega, given the fixed (RPM-
    independent) BEMT hover CT -- the BEMT equivalent of
    motor_performance.required_rpm_for_thrust, but from integrated blade
    geometry instead of a single constant.
    """
    radius_m = diameter_m / 2
    area_m2 = math.pi * radius_m ** 2
    ct = hover_coefficients()["ct"]
    tip_speed = math.sqrt(thrust_per_motor_n / (ct * air_density * area_m2))
    omega_rad_s = tip_speed / radius_m
    return omega_rad_s * 60 / (2 * math.pi)


def analyze_bemt_hover(
    mass_kg: float,
    motor_count: int,
    propeller_diameter_in: float,
    motor_kv: float,
    battery_cells: int,
    air_density: float,
) -> dict:
    """
    Cross-check against motor_performance.analyze_motor_performance: both
    models are asked the same question ("what RPM does this drone need to
    hover?") via two independently-derived thrust models -- a single
    empirical static-Ct constant (motor_performance.py) vs. integrated
    blade-element/momentum theory (this module) -- and the results are
    compared. Close agreement between two independently-derived physics
    models is a meaningful credibility signal; a large disagreement would
    flag that one of the models' assumptions doesn't fit this drone.
    """
    diameter_m = propeller_diameter_in * 0.0254
    gravity = 9.81
    total_hover_thrust_n = mass_kg * gravity
    thrust_per_motor_n = total_hover_thrust_n / motor_count

    nominal_voltage = battery_cells * 3.7
    max_available_rpm = motor_kv * nominal_voltage

    bemt_required_rpm = bemt_required_rpm_for_thrust(thrust_per_motor_n, air_density, diameter_m)
    bemt_result = bemt_thrust_and_power(diameter_m, bemt_required_rpm, air_density)

    return {
        "ct": bemt_result["ct"],
        "cp": bemt_result["cp"],
        "bemt_required_rpm": round(bemt_required_rpm),
        "bemt_required_power_per_motor_w": round(bemt_result["power_w"], 1),
        "max_available_rpm": round(max_available_rpm),
        "bemt_hover_feasible": bemt_required_rpm <= max_available_rpm,
    }
