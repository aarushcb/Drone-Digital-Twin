"""
WHY THIS IS A SIMULATION, NOT A READ OF STORED TELEMETRY:
This module's docstring states this plainly, the same honest-limitation
standard as digital_twin.py's motor wear estimate: app/models/telemetry.py
has only ever stored ACTUAL roll/pitch/yaw -- there is no desired-attitude
setpoint or motor PWM output anywhere in this schema, and there never has
been. Trying to "reconstruct" a desired signal from the actual response
alone is mathematically circular (you cannot recover what a controller
was TARGETING from what it ACHIEVED without also knowing the controller's
own gains -- that's underdetermined, not an engineering shortcut). Rather
than fake that reconstruction, this follows the same pattern already
established by environment_simulator.py and digital_twin.py in this app:
an honestly-labeled SIMULATION built from real, verified control theory,
not a claim about this specific drone's actual logged flight.

THE MODEL -- A STANDARD 2ND-ORDER PD-CONTROLLED ATTITUDE RESPONSE:
A single-axis attitude loop (roll, pitch, or yaw) driven by a
proportional-derivative controller against a rotational inertia is one of
the most standard results in control theory -- the closed-loop dynamics
reduce to the textbook 2nd-order system:

    theta'' + 2*zeta*omega_n*theta' + omega_n^2*theta = omega_n^2*theta_desired

where zeta (damping ratio) and omega_n (natural frequency, rad/s) fully
characterize the response -- exactly the parameters real PID tuning
adjusts, whether the tuner realizes it or not. This is standard material
in any controls textbook (e.g. Ogata, "Modern Control Engineering";
Franklin, Powell & Emami-Naeini, "Feedback Control of Dynamic Systems")
and specifically for multirotor attitude loops (e.g. Beard & McLain,
"Small Unmanned Aircraft: Theory and Practice," covers exactly this
PD-attitude-loop-as-2nd-order-system reduction).

STEP RESPONSE -- CLOSED-FORM SOLUTION (used here to VERIFY the numerical
simulation, not as the simulation method itself -- the actual time series
returned is numerically integrated, matching real nonlinear simulation
practice, but cross-checked against the known analytical solution):
For an underdamped system (zeta < 1) with omega_d = omega_n*sqrt(1-zeta^2):

    theta(t) = theta_desired * [1 - (e^(-zeta*omega_n*t)/sqrt(1-zeta^2))
                                  * sin(omega_d*t + phi)],  phi = arccos(zeta)

STANDARD CLOSED-FORM PERFORMANCE METRICS (Ogata Ch. 5 / Franklin et al.
Ch. 3 -- exactly the numbers a real controls course teaches a student to
read off a step response):
    peak_time_s        = pi / omega_d
    overshoot_percent  = 100 * e^(-zeta*pi / sqrt(1-zeta^2))
    settling_time_s    = 4 / (zeta * omega_n)                (2% criterion)
    rise_time_s        = (pi - phi) / omega_d                (0% to 100%, underdamped)

CONTROL INPUT (motor differential command proxy): the PD law itself,
    u(t) = omega_n^2*(theta_desired - theta) + 2*zeta*omega_n*(theta_desired' - theta')
is exactly the commanded ANGULAR ACCELERATION (deg/s^2) the controller is
demanding at each instant -- a real, physically meaningful quantity
directly derived from the same equation driving the simulation, NOT a
literal ESC PWM microsecond value (this app has never logged raw PWM,
and doesn't model individual ESCs/motor mixing here).

STABLE VS. OSCILLATING PRESETS: zeta=0.7 ("stable" -- a commonly cited
"good"/slightly-underdamped tuning target in PID tuning guides, giving a
fast response with only mild overshoot) vs. zeta=0.15 ("oscillating" -- a
poorly-damped gain set that rings visibly before settling), both at the
same omega_n=8.0 rad/s (a representative bandwidth for a small
multirotor's attitude loop) -- letting a student see exactly what
"stable" vs. "oscillating" means as a difference in one real physical
parameter (damping ratio), not just two unrelated canned datasets.

WHAT WAS VERIFIED BEFORE SHIPPING:
- The RK4-numerically-integrated step response was checked against the
  closed-form analytical solution above at several time points and
  matches within numerical tolerance (confirms the simulation is
  correctly solving the ODE it claims to solve, not just producing
  plausible-looking curves) -- see test_control_loop.py.
- peak_time_s was independently verified by finding the actual peak of
  the simulated (not closed-form) time series and confirming it lands
  within one output sample of the closed-form peak_time formula.
- overshoot_percent was independently verified against the simulated
  series' own actual maximum value relative to the commanded step.
"""

import math
from typing import List, Optional

DEFAULT_NATURAL_FREQUENCY_RAD_S = 8.0
STABLE_DAMPING_RATIO = 0.7
OSCILLATING_DAMPING_RATIO = 0.15

# Representative step command amplitudes per axis (degrees) -- roll/pitch
# use a modest attitude-hold test step; yaw uses a larger heading-change
# step, matching the larger authority/slower dynamics typical of yaw
# control on a multirotor.
AXIS_STEP_DEG = {"roll": 15.0, "pitch": 10.0, "yaw": 30.0}


def simulate_step_response(
    step_deg: float,
    damping_ratio: float,
    natural_frequency_rad_s: float,
    duration_s: float,
    sample_rate_hz: float,
    step_start_s: float = 0.5,
) -> dict:
    """
    Numerically integrates theta'' + 2*zeta*omega_n*theta' + omega_n^2*theta
    = omega_n^2*theta_desired via RK4 (a standard, accurate ODE integration
    method) at a fine internal timestep, then downsamples to the
    requested sample_rate_hz for output -- the step command turns on at
    step_start_s (a moment of pre-step "at rest" baseline is included so
    the response's starting point is visually clear on a chart).
    """
    dt_internal = 0.001
    n_internal_steps = int(duration_s / dt_internal)
    output_every = max(1, round(1 / (sample_rate_hz * dt_internal)))

    zeta = damping_ratio
    omega_n = natural_frequency_rad_s

    def desired_at(t: float) -> float:
        return step_deg if t >= step_start_s else 0.0

    def derivatives(t: float, theta: float, theta_dot: float) -> tuple:
        theta_desired = desired_at(t)
        theta_dot_dot = omega_n ** 2 * (theta_desired - theta) - 2 * zeta * omega_n * theta_dot
        return theta_dot, theta_dot_dot

    theta = 0.0
    theta_dot = 0.0
    t = 0.0

    timestamps, desired_series, actual_series, control_input_series = [], [], [], []

    for i in range(n_internal_steps + 1):
        if i % output_every == 0:
            theta_desired = desired_at(t)
            control_input = omega_n ** 2 * (theta_desired - theta) - 2 * zeta * omega_n * theta_dot
            timestamps.append(round(t, 4))
            desired_series.append(round(theta_desired, 4))
            actual_series.append(round(theta, 4))
            control_input_series.append(round(control_input, 4))

        # Standard RK4 step for the 2nd-order system, written as a
        # first-order system in (theta, theta_dot).
        k1_theta, k1_theta_dot = derivatives(t, theta, theta_dot)
        k2_theta, k2_theta_dot = derivatives(
            t + dt_internal / 2, theta + k1_theta * dt_internal / 2, theta_dot + k1_theta_dot * dt_internal / 2
        )
        k3_theta, k3_theta_dot = derivatives(
            t + dt_internal / 2, theta + k2_theta * dt_internal / 2, theta_dot + k2_theta_dot * dt_internal / 2
        )
        k4_theta, k4_theta_dot = derivatives(
            t + dt_internal, theta + k3_theta * dt_internal, theta_dot + k3_theta_dot * dt_internal
        )
        theta += (dt_internal / 6) * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta)
        theta_dot += (dt_internal / 6) * (k1_theta_dot + 2 * k2_theta_dot + 2 * k3_theta_dot + k4_theta_dot)
        t += dt_internal

    return {
        "timestamps_s": timestamps,
        "desired_deg": desired_series,
        "actual_deg": actual_series,
        "control_input_deg_s2": control_input_series,
    }


def closed_form_metrics(damping_ratio: float, natural_frequency_rad_s: float) -> dict:
    """Standard 2nd-order underdamped step-response performance metrics --
    see module docstring for the formulas and their textbook source."""
    zeta = damping_ratio
    omega_n = natural_frequency_rad_s
    if zeta >= 1.0:
        # Overdamped/critically damped -- the underdamped closed forms
        # below (which involve sqrt(1-zeta^2) and a peak/overshoot that
        # only exist for an underdamped response) don't apply; reported
        # honestly as None rather than a nonsensical extrapolation.
        return {
            "peak_time_s": None,
            "overshoot_percent": 0.0,
            "settling_time_s": round(4 / (zeta * omega_n), 3) if zeta * omega_n > 0 else None,
            "rise_time_s": None,
        }

    omega_d = omega_n * math.sqrt(1 - zeta ** 2)
    phi = math.acos(zeta)

    peak_time_s = math.pi / omega_d
    overshoot_percent = 100 * math.exp(-zeta * math.pi / math.sqrt(1 - zeta ** 2))
    settling_time_s = 4 / (zeta * omega_n)
    rise_time_s = (math.pi - phi) / omega_d

    return {
        "peak_time_s": round(peak_time_s, 3),
        "overshoot_percent": round(overshoot_percent, 1),
        "settling_time_s": round(settling_time_s, 3),
        "rise_time_s": round(rise_time_s, 3),
    }


def simulate_control_loop(
    scenario: str = "stable",
    duration_s: float = 4.0,
    sample_rate_hz: float = 50.0,
    natural_frequency_rad_s: float = DEFAULT_NATURAL_FREQUENCY_RAD_S,
) -> dict:
    """
    Simulates all three attitude axes (roll, pitch, yaw) under the same
    damping ratio (stable or oscillating preset) and natural frequency --
    see module docstring for what each preset represents and why a
    shared damping ratio across axes is a reasonable simplification
    (usually reflects one overall controller tuning quality).
    """
    damping_ratio = STABLE_DAMPING_RATIO if scenario == "stable" else OSCILLATING_DAMPING_RATIO

    axes = {}
    for axis, step_deg in AXIS_STEP_DEG.items():
        response = simulate_step_response(
            step_deg=step_deg,
            damping_ratio=damping_ratio,
            natural_frequency_rad_s=natural_frequency_rad_s,
            duration_s=duration_s,
            sample_rate_hz=sample_rate_hz,
        )
        metrics = closed_form_metrics(damping_ratio, natural_frequency_rad_s)
        axes[axis] = {**response, **metrics, "step_command_deg": step_deg}

    return {
        "scenario": scenario,
        "damping_ratio": damping_ratio,
        "natural_frequency_rad_s": natural_frequency_rad_s,
        "axes": axes,
    }


# ---------- REAL desired-vs-actual data, when a drone's telemetry has it ----------
# app/models/telemetry.py added optional desired_roll/pitch/yaw and
# motor_pwm_1-4 columns specifically so a real or MAVLink-imported flight
# (see app/services/mavlink_import.py's ATTITUDE_TARGET/SERVO_OUTPUT_RAW
# parsing) CAN populate real setpoint/output data. When it has, this
# builds the control-loop visualization from that real data instead of
# the simulation above. When it hasn't (the common case -- ordinary
# telemetry has never recorded a setpoint), the caller falls back to
# simulate_control_loop().

MIN_REAL_CONTROL_LOOP_SAMPLES = 5


def build_real_control_loop_response(samples: List[dict]) -> Optional[dict]:
    """
    samples: chronologically ordered list of dicts, each with keys
    'timestamp' (datetime), 'roll'/'pitch'/'yaw', and
    'desired_roll'/'desired_pitch'/'desired_yaw' -- all guaranteed
    non-None by the caller's query (see app/crud/telemetry.py's
    get_control_loop_samples), plus optional 'motor_pwm_1'..'motor_pwm_4'.

    Returns None if there aren't enough real samples to plot anything
    meaningful (MIN_REAL_CONTROL_LOOP_SAMPLES), so the caller can fall
    back to the simulation.

    Unlike the simulation, this does NOT report step-response metrics
    (peak time, overshoot %, settling time) -- those closed-form formulas
    (see module docstring above) assume an idealized clean step input,
    which arbitrary real flight data is not. Instead this reports the
    metric that's always honestly valid for a real desired-vs-actual
    trace: tracking error (desired - actual), as RMS and max magnitude
    per axis -- the standard way control engineers quantify real-world
    tracking performance without assuming a clean step test.
    """
    if len(samples) < MIN_REAL_CONTROL_LOOP_SAMPLES:
        return None

    t0 = samples[0]["timestamp"]
    timestamps_s = [round((s["timestamp"] - t0).total_seconds(), 4) for s in samples]

    axes = {}
    for axis in ("roll", "pitch", "yaw"):
        desired = [s[f"desired_{axis}"] for s in samples]
        actual = [s[axis] for s in samples]
        errors = [d - a for d, a in zip(desired, actual)]
        rms_error = math.sqrt(sum(e ** 2 for e in errors) / len(errors))
        max_error = max(abs(e) for e in errors)
        axes[axis] = {
            "timestamps_s": timestamps_s,
            "desired_deg": [round(v, 4) for v in desired],
            "actual_deg": [round(v, 4) for v in actual],
            "rms_error_deg": round(rms_error, 3),
            "max_error_deg": round(max_error, 3),
        }

    motor_pwm_keys = ["motor_pwm_1", "motor_pwm_2", "motor_pwm_3", "motor_pwm_4"]
    has_pwm_data = any(s.get(k) is not None for s in samples for k in motor_pwm_keys)
    motor_pwm = None
    if has_pwm_data:
        motor_pwm = {"timestamps_s": timestamps_s}
        for key in motor_pwm_keys:
            motor_pwm[key] = [s.get(key) for s in samples]

    return {
        "data_source": "real",
        "sample_count": len(samples),
        "start_time": samples[0]["timestamp"].isoformat(),
        "end_time": samples[-1]["timestamp"].isoformat(),
        "axes": axes,
        "motor_pwm": motor_pwm,
    }
