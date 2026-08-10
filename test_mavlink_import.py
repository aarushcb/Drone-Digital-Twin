"""
Standalone test script for the MAVLink import quaternion-to-Euler
conversion (used to turn ATTITUDE_TARGET's attitude quaternion into
desired_roll/pitch/yaw degrees -- see app/services/mavlink_import.py and
app/models/telemetry.py). Run directly with `python3 test_mavlink_import.py`,
matching the other standalone test_*.py scripts in this repo.

Building a full synthetic .tlog to test parse_mavlink_log() end-to-end is
out of scope here (it needs pymavlink's binary log writer) -- this tests
the actual new math this feature added: the quaternion conversion, against
known reference rotations that have an unambiguous expected answer.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))

from app.services.mavlink_import import _quaternion_to_euler_deg


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


if __name__ == "__main__":
    test_identity_quaternion_is_zero_attitude()
    test_90deg_yaw_rotation_quaternion()
    test_45deg_roll_rotation_quaternion()
    test_30deg_pitch_rotation_quaternion()
    print("\nAll MAVLink import tests passed.")
