"""
Standalone test script for MAVLink import's unit conversions (quaternion
-> Euler for ATTITUDE_TARGET, mG/mrad-per-s -> m/s^2/deg-per-s for
SCALED_IMU/RAW_IMU) plus an end-to-end synthetic-log parser test for the
IMU import path -- see app/services/mavlink_import.py and
app/models/telemetry.py. Run directly with `python3 test_mavlink_import.py`,
matching the other standalone test_*.py scripts in this repo.
"""
import struct
import sys, os, math, time
sys.path.insert(0, os.path.dirname(__file__))

from pymavlink import mavutil

from app.services.mavlink_import import (
    _quaternion_to_euler_deg, _accel_mg_to_mps2, _gyro_mrad_s_to_deg_s,
    parse_mavlink_log, GRAVITY_MPS2,
)


def test_identity_quaternion_is_zero_attitude():
    roll, pitch, yaw = _quaternion_to_euler_deg([1.0, 0.0, 0.0, 0.0])
    assert abs(roll) < 1e-9 and abs(pitch) < 1e-9 and abs(yaw) < 1e-9
    print(f"PASS: identity quaternion -> roll={roll}, pitch={pitch}, yaw={yaw}")


def test_90deg_yaw_rotation_quaternion():
    # Rotation of +90deg about the z axis: q = (cos(45deg), 0, 0, sin(45deg)).
    half = math.radians(45)
    q = [math.cos(half), 0.0, 0.0, math.sin(half)]
    roll, pitch, yaw = _quaternion_to_euler_deg(q)
    assert abs(roll) < 1e-6, roll
    assert abs(pitch) < 1e-6, pitch
    assert abs(yaw - 90.0) < 1e-6, yaw
    print(f"PASS: 90deg yaw-rotation quaternion -> yaw={yaw} (roll={roll}, pitch={pitch})")


def test_45deg_roll_rotation_quaternion():
    # Rotation of +45deg about the x axis: q = (cos(22.5deg), sin(22.5deg), 0, 0).
    half = math.radians(22.5)
    q = [math.cos(half), math.sin(half), 0.0, 0.0]
    roll, pitch, yaw = _quaternion_to_euler_deg(q)
    assert abs(roll - 45.0) < 1e-6, roll
    assert abs(pitch) < 1e-6, pitch
    assert abs(yaw) < 1e-6, yaw
    print(f"PASS: 45deg roll-rotation quaternion -> roll={roll} (pitch={pitch}, yaw={yaw})")


def test_30deg_pitch_rotation_quaternion():
    # Rotation of +30deg about the y axis: q = (cos(15deg), 0, sin(15deg), 0).
    half = math.radians(15)
    q = [math.cos(half), 0.0, math.sin(half), 0.0]
    roll, pitch, yaw = _quaternion_to_euler_deg(q)
    assert abs(roll) < 1e-6, roll
    assert abs(pitch - 30.0) < 1e-6, pitch
    assert abs(yaw) < 1e-6, yaw
    print(f"PASS: 30deg pitch-rotation quaternion -> pitch={pitch} (roll={roll}, yaw={yaw})")


def test_accel_mg_to_mps2_matches_known_reference():
    # 1000 mG = 1 standard gravity, by definition -- should convert to
    # exactly GRAVITY_MPS2 (9.81), the same constant this app uses
    # everywhere else for standard gravity.
    assert abs(_accel_mg_to_mps2(1000.0) - GRAVITY_MPS2) < 1e-9
    assert abs(_accel_mg_to_mps2(0.0) - 0.0) < 1e-9
    assert abs(_accel_mg_to_mps2(-1000.0) - (-GRAVITY_MPS2)) < 1e-9
    print(f"PASS: 1000mG -> {_accel_mg_to_mps2(1000.0)}m/s^2 (exactly GRAVITY_MPS2={GRAVITY_MPS2})")


def test_gyro_mrad_s_to_deg_s_matches_known_reference():
    # 1000 mrad/s = 1 rad/s -- a well-known reference conversion,
    # 1 rad = 180/pi deg ~= 57.2958 deg.
    result = _gyro_mrad_s_to_deg_s(1000.0)
    expected = 180.0 / math.pi
    assert abs(result - expected) < 1e-6, f"expected {expected}, got {result}"
    print(f"PASS: 1000mrad/s -> {result:.4f}deg/s (matches known reference 1rad/s={expected:.4f}deg/s)")


def _write_test_tlog(path: str, imu_samples: list, hdg: int = 9000):
    """
    Builds a small synthetic .tlog (same wire format used by
    generate_test_flight_log.py) with a disarmed HEARTBEAT, one
    SCALED_IMU message per entry in imu_samples (each a
    (xacc_mg, yacc_mg, zacc_mg, xgyro_mrad, ygyro_mrad, zgyro_mrad)
    tuple), and one GLOBAL_POSITION_INT sync point after each -- enough
    to exercise parse_mavlink_log()'s IMU forward-fill end to end without
    touching the committed test_flight_synthetic.tlog fixture (which
    several other scripts depend on having an exact, unchanged shape).
    """
    mav = mavutil.mavlink.MAVLink(None, srcSystem=1, srcComponent=1)
    base_wall_time = time.time()

    with open(path, "wb") as f:
        def write_msg(msg, t_seconds):
            tusec = int((base_wall_time + t_seconds) * 1e6)
            f.write(struct.pack(">Q", tusec))
            f.write(msg.pack(mav))

        for i, (xacc, yacc, zacc, xgyro, ygyro, zgyro) in enumerate(imu_samples):
            t = float(i)
            # Disarmed throughout -- base_mode=0 (no MAV_MODE_FLAG_SAFETY_ARMED
            # bit), so parse_mavlink_log reports flight_state="idle", the
            # correct state for a stationary calibration-representative sample.
            write_msg(mav.heartbeat_encode(type=2, autopilot=3, base_mode=0, custom_mode=0, system_status=3), t)
            write_msg(mav.scaled_imu_encode(
                time_boot_ms=int(t * 1000),
                xacc=xacc, yacc=yacc, zacc=zacc,
                xgyro=xgyro, ygyro=ygyro, zgyro=zgyro,
                xmag=0, ymag=0, zmag=0,
            ), t)
            write_msg(mav.global_position_int_encode(
                time_boot_ms=int(t * 1000),
                lat=int(37.7749 * 1e7), lon=int(-122.4194 * 1e7),
                alt=0, relative_alt=0, vx=0, vy=0, vz=0, hdg=hdg,
            ), t)


def test_scaled_imu_parses_and_converts_end_to_end():
    # A real, complete parser test (not just the conversion function in
    # isolation): builds a genuine synthetic .tlog with SCALED_IMU
    # messages carrying known mG/mrad-per-s values, runs it through the
    # actual parse_mavlink_log(), and confirms the emitted points carry
    # the correctly-converted m/s^2/deg-per-s values -- and that a
    # disarmed HEARTBEAT correctly yields flight_state="idle" (the state
    # the Calibration Guide's real-data path specifically looks for).
    path = "/tmp/test_scaled_imu_sample.tlog"
    # (xacc, yacc, zacc, xgyro, ygyro, zgyro) in (mG, mG, mG, mrad/s, mrad/s, mrad/s)
    # -- representative "sitting still, level" values: near-zero x/y accel,
    # ~1G on z, near-zero rotation on all axes.
    samples = [
        (5, -3, 998, 2, -1, 0),
        (-2, 4, 1002, -1, 1, 1),
        (1, -1, 1000, 0, 0, -1),
    ]
    try:
        _write_test_tlog(path, samples)
        points = parse_mavlink_log(path)
        assert len(points) == len(samples), f"expected {len(samples)} points, got {len(points)}"

        for point, (xacc, yacc, zacc, xgyro, ygyro, zgyro) in zip(points, samples):
            assert abs(point.accel_x - _accel_mg_to_mps2(xacc)) < 1e-9
            assert abs(point.accel_y - _accel_mg_to_mps2(yacc)) < 1e-9
            assert abs(point.accel_z - _accel_mg_to_mps2(zacc)) < 1e-9
            assert abs(point.gyro_x - _gyro_mrad_s_to_deg_s(xgyro)) < 1e-9
            assert abs(point.gyro_y - _gyro_mrad_s_to_deg_s(ygyro)) < 1e-9
            assert abs(point.gyro_z - _gyro_mrad_s_to_deg_s(zgyro)) < 1e-9
            assert point.flight_state == "idle", f"expected idle (disarmed), got {point.flight_state}"

        last = points[-1]
        print(
            f"PASS: {len(points)} SCALED_IMU points parsed and converted correctly end-to-end "
            f"(last point: accel=({last.accel_x:.3f},{last.accel_y:.3f},{last.accel_z:.3f})m/s^2, "
            f"gyro=({last.gyro_x:.3f},{last.gyro_y:.3f},{last.gyro_z:.3f})deg/s, "
            f"flight_state={last.flight_state})"
        )
    finally:
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    test_identity_quaternion_is_zero_attitude()
    test_90deg_yaw_rotation_quaternion()
    test_45deg_roll_rotation_quaternion()
    test_30deg_pitch_rotation_quaternion()
    test_accel_mg_to_mps2_matches_known_reference()
    test_gyro_mrad_s_to_deg_s_matches_known_reference()
    test_scaled_imu_parses_and_converts_end_to_end()
    print("\nAll MAVLink import tests passed.")
