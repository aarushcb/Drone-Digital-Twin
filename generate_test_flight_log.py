"""
Generates a small SYNTHETIC MAVLink .tlog flight log with a known,
hand-calculable flight profile, for testing the MAVLink import feature
(app/services/mavlink_import.py, POST /drones/{id}/import-mavlink)
end-to-end -- including the recently-added desired-attitude/motor-PWM
fields (ATTITUDE_TARGET, SERVO_OUTPUT_RAW) that feed the control-loop
"real data" path (app/services/control_loop.py).

Run with: python3 generate_test_flight_log.py
Output: test_flight_synthetic.tlog in the current directory (~1-2KB).

THE FLIGHT PROFILE (t = whole seconds, t=0..130, 131 points total --
one GLOBAL_POSITION_INT per second, which is the sync point
mavlink_import.py emits one merged telemetry row per):

  TAKEOFF  t in [0, 20]   (21 points): vertical climb, home position held
  CRUISE   t in [21, 110] (90 points): level flight heading due east
  LANDING  t in [111, 130] (20 points): vertical descent, no horizontal motion

Every value below is a DETERMINISTIC, closed-form function of t --
after importing this log, each imported telemetry row's timestamp maps
back to a whole-second t (row 0 = t=0, row 1 = t=1, ...), and every
field can be independently recomputed and checked against what the app
shows. All formulas are simple ON PURPOSE (piecewise-linear, constant
per phase) so they're easy to recompute by hand, not because they're
claimed to be aerodynamically realistic.

ALTITUDE (relative, meters) -- linear climb/descent, flat cruise:
    t <= 20:            altitude(t) = 1.5 * t              (0m -> 30m)
    20 < t <= 110:       altitude(t) = 30.0                 (flat)
    t > 110:             altitude(t) = 30.0 - 1.5*(t-110)   (30m -> 0m, exactly 0 at t=130)

BATTERY (percent) -- linear drain at a different (documented) rate per
phase (climb draws more power than cruise; descent a bit more than
cruise, less than climb):
    t <= 20:             battery(t) = 100.0 - 0.15*t             (100.0% -> 97.0%)
    20 < t <= 110:        battery(t) = 97.0 - 0.08*(t-20)         (97.0% -> 89.8%)
    t > 110:              battery(t) = 89.8 - 0.10*(t-110)        (89.8% -> 87.8%)

GROUNDSPEED (m/s) / HORIZONTAL POSITION -- stationary during vertical
climb/descent, constant 8 m/s due east during cruise:
    t <= 20 or t > 110:   groundspeed(t) = 0.0
    20 < t <= 110:         groundspeed(t) = 8.0
    distance_east(t) (meters) = 8.0 * max(0, min(t, 110) - 20)
    -> 0m for t<=20, growing to exactly 720.0m at t=110, held at 720.0m during landing.
    Longitude offset from home: delta_lon_deg = distance_east(t) / (111320 * cos(radians(HOME_LAT)))
    (standard "meters per degree longitude at this latitude" approximation)

ATTITUDE (degrees) -- nose-up during climb, nose-down during descent,
level and heading due east (yaw=90) throughout, no banking:
    roll(t)  = 0.0 always
    yaw(t)   = 90.0 always
    pitch(t) = -5.0 (t<=20, climb) | 0.0 (20<t<=110, cruise) | +5.0 (t>110, descent)

DESIRED ATTITUDE (ATTITUDE_TARGET, imported as desired_roll/pitch/yaw) --
set to a constant +1.5deg roll offset from the actual roll, desired
pitch/yaw exactly equal to actual, at every single point. This makes the
control-loop "real data" tracking-error check trivially verifiable:
after import, GET /drones/{id}/telemetry/control-loop should report
rms_error_deg=1.5 and max_error_deg=1.5 for roll, and 0.0 for pitch/yaw.
(ATTITUDE_TARGET transmits this as a quaternion, not raw Euler angles --
generated here via the exact mathematical inverse of the quaternion-to-
Euler conversion in mavlink_import.py, and round-trip-verified against
that exact function before this file was generated -- see this script's
own self-test output.)

MOTOR PWM (SERVO_OUTPUT_RAW, microseconds, all 4 motors equal -- no
attitude-hold differential is modeled, this is a synthetic level-flight
log) -- higher for climb, nominal for cruise, lower for descent:
    t <= 20:              1650us (climb -- above-hover throttle)
    20 < t <= 110:         1550us (cruise -- roughly hover throttle)
    t > 110:               1450us (descent -- below-hover throttle)

flight_state: "flying" for every point (armed throughout this log).
"""
import math
import struct
import time

from pymavlink import mavutil

OUTPUT_PATH = "test_flight_synthetic.tlog"

HOME_LAT = 37.7749
HOME_LON = -122.4194
BASE_WALL_TIME = time.time()  # tlog timestamps are wall-clock; only the RELATIVE spacing matters for verification

T_TAKEOFF_END = 20
T_CRUISE_END = 110
T_LANDING_END = 130

CLIMB_RATE_MPS = 1.5
CRUISE_GROUNDSPEED_MPS = 8.0

BATTERY_DRAIN_TAKEOFF_PCT_PER_S = 0.15
BATTERY_DRAIN_CRUISE_PCT_PER_S = 0.08
BATTERY_DRAIN_LANDING_PCT_PER_S = 0.10

PITCH_CLIMB_DEG = -5.0
PITCH_CRUISE_DEG = 0.0
PITCH_LANDING_DEG = 5.0
YAW_DEG = 90.0
ROLL_DEG = 0.0
DESIRED_ROLL_OFFSET_DEG = 1.5

PWM_CLIMB_US = 1650
PWM_CRUISE_US = 1550
PWM_LANDING_US = 1450


def euler_to_quaternion(roll_deg: float, pitch_deg: float, yaw_deg: float):
    """Exact mathematical inverse of the ZYX Tait-Bryan quaternion-to-
    Euler conversion in app/services/mavlink_import.py's
    _quaternion_to_euler_deg -- round-trip-verified against that exact
    function (see this module's __main__ self-test) before this
    generator was used to produce the shipped test log."""
    r, p, y = math.radians(roll_deg) / 2, math.radians(pitch_deg) / 2, math.radians(yaw_deg) / 2
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y_ = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [w, x, y_, z]


def altitude_m(t: int) -> float:
    if t <= T_TAKEOFF_END:
        return CLIMB_RATE_MPS * t
    if t <= T_CRUISE_END:
        return 30.0
    return 30.0 - CLIMB_RATE_MPS * (t - T_CRUISE_END)


def battery_pct(t: int) -> float:
    if t <= T_TAKEOFF_END:
        return 100.0 - BATTERY_DRAIN_TAKEOFF_PCT_PER_S * t
    if t <= T_CRUISE_END:
        return 97.0 - BATTERY_DRAIN_CRUISE_PCT_PER_S * (t - T_TAKEOFF_END)
    return 89.8 - BATTERY_DRAIN_LANDING_PCT_PER_S * (t - T_CRUISE_END)


def groundspeed_mps(t: int) -> float:
    return CRUISE_GROUNDSPEED_MPS if T_TAKEOFF_END < t <= T_CRUISE_END else 0.0


def distance_east_m(t: int) -> float:
    return CRUISE_GROUNDSPEED_MPS * max(0, min(t, T_CRUISE_END) - T_TAKEOFF_END)


def pitch_deg(t: int) -> float:
    if t <= T_TAKEOFF_END:
        return PITCH_CLIMB_DEG
    if t <= T_CRUISE_END:
        return PITCH_CRUISE_DEG
    return PITCH_LANDING_DEG


def motor_pwm_us(t: int) -> int:
    if t <= T_TAKEOFF_END:
        return PWM_CLIMB_US
    if t <= T_CRUISE_END:
        return PWM_CRUISE_US
    return PWM_LANDING_US


def generate():
    mav = mavutil.mavlink.MAVLink(None, srcSystem=1, srcComponent=1)

    with open(OUTPUT_PATH, "wb") as f:
        def write_msg(msg, t_seconds: int):
            wall_time = BASE_WALL_TIME + t_seconds
            tusec = int(wall_time * 1e6)
            f.write(struct.pack(">Q", tusec))
            f.write(msg.pack(mav))

        for t in range(0, T_LANDING_END + 1):
            time_boot_ms = t * 1000

            # ARMED throughout (bit 7 / value 128 of base_mode -- see
            # mavlink_import.py's HEARTBEAT handling).
            heartbeat = mav.heartbeat_encode(
                type=2,  # MAV_TYPE_QUADROTOR
                autopilot=3,  # MAV_AUTOPILOT_ARDUPILOTMEGA
                base_mode=128,
                custom_mode=0,
                system_status=4,  # MAV_STATE_ACTIVE
            )
            write_msg(heartbeat, t)

            roll, pitch, yaw = ROLL_DEG, pitch_deg(t), YAW_DEG
            attitude = mav.attitude_encode(
                time_boot_ms=time_boot_ms,
                roll=math.radians(roll), pitch=math.radians(pitch), yaw=math.radians(yaw),
                rollspeed=0.0, pitchspeed=0.0, yawspeed=0.0,
            )
            write_msg(attitude, t)

            desired_q = euler_to_quaternion(roll + DESIRED_ROLL_OFFSET_DEG, pitch, yaw)
            attitude_target = mav.attitude_target_encode(
                time_boot_ms=time_boot_ms,
                type_mask=0,
                q=desired_q,
                body_roll_rate=0.0, body_pitch_rate=0.0, body_yaw_rate=0.0,
                thrust=0.5,
            )
            write_msg(attitude_target, t)

            battery = battery_pct(t)
            sys_status = mav.sys_status_encode(
                onboard_control_sensors_present=0, onboard_control_sensors_enabled=0,
                onboard_control_sensors_health=0, load=0,
                voltage_battery=14800, current_battery=1500,
                battery_remaining=round(battery),
                drop_rate_comm=0, errors_comm=0,
                errors_count1=0, errors_count2=0, errors_count3=0, errors_count4=0,
            )
            write_msg(sys_status, t)

            speed = groundspeed_mps(t)
            vfr_hud = mav.vfr_hud_encode(
                airspeed=speed, groundspeed=speed, heading=int(yaw),
                throttle=50, alt=altitude_m(t), climb=0.0,
            )
            write_msg(vfr_hud, t)

            pwm = motor_pwm_us(t)
            servo_output = mav.servo_output_raw_encode(
                time_usec=time_boot_ms * 1000, port=0,
                servo1_raw=pwm, servo2_raw=pwm, servo3_raw=pwm, servo4_raw=pwm,
                servo5_raw=0, servo6_raw=0, servo7_raw=0, servo8_raw=0,
            )
            write_msg(servo_output, t)

            # Sync point -- emits one merged telemetry row (see
            # mavlink_import.py's docstring).
            alt = altitude_m(t)
            dist_east = distance_east_m(t)
            delta_lon_deg = dist_east / (111320.0 * math.cos(math.radians(HOME_LAT)))
            lat = HOME_LAT
            lon = HOME_LON + delta_lon_deg

            global_position = mav.global_position_int_encode(
                time_boot_ms=time_boot_ms,
                lat=round(lat * 1e7), lon=round(lon * 1e7),
                alt=round(alt * 1000), relative_alt=round(alt * 1000),
                vx=round(speed * 100), vy=0, vz=0,
                hdg=round(yaw * 100),
            )
            write_msg(global_position, t)

    print(f"Wrote {OUTPUT_PATH} ({T_LANDING_END + 1} points, t=0..{T_LANDING_END}s)")


def self_test_quaternion_round_trip():
    """Confirms euler_to_quaternion is the exact inverse of the app's own
    _quaternion_to_euler_deg before this generator is trusted to produce
    a verifiable log -- run automatically every time this script runs."""
    from app.services.mavlink_import import _quaternion_to_euler_deg

    test_cases = [
        (0.0, 0.0, 0.0),
        (ROLL_DEG + DESIRED_ROLL_OFFSET_DEG, PITCH_CLIMB_DEG, YAW_DEG),
        (ROLL_DEG + DESIRED_ROLL_OFFSET_DEG, PITCH_CRUISE_DEG, YAW_DEG),
        (ROLL_DEG + DESIRED_ROLL_OFFSET_DEG, PITCH_LANDING_DEG, YAW_DEG),
    ]
    for roll, pitch, yaw in test_cases:
        q = euler_to_quaternion(roll, pitch, yaw)
        r2, p2, y2 = _quaternion_to_euler_deg(q)
        y2 = y2 % 360
        assert abs(r2 - roll) < 1e-6, (roll, r2)
        assert abs(p2 - pitch) < 1e-6, (pitch, p2)
        assert abs(y2 - (yaw % 360)) < 1e-6, (yaw, y2)
    print("Self-test passed: euler_to_quaternion round-trips exactly through "
          "the app's own _quaternion_to_euler_deg for every angle combination used in this log.")


if __name__ == "__main__":
    self_test_quaternion_round_trip()
    generate()
