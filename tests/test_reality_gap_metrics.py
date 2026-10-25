"""
D1 pass-1 tests -- PARSER AND ALIGNMENT ONLY.

Scope note: this file is named test_reality_gap_metrics.py because that is
the filename the spec's section 12 manifest assigns it, but D1 pass 1
built only the ingestion half of the pipeline (parsers.py + align.py).
The metric-correctness cases of spec 10.1 (KL against a closed form, KS
against scipy, score monotonicity, ...) are NOT here because metrics.py
does not exist yet -- they arrive with it, in the same file.

Run:  python3 tests/test_reality_gap_metrics.py

WHY A STANDALONE SCRIPT RATHER THAN pytest: spec 10.1 says "unit tests,
pytest", but this repo has no pytest installed and every existing
test_*.py is a standalone script run directly (see CLAUDE.md, "Testing
tools in the backend repo"). Adding pytest purely for this module would
be a new production dependency for no behavioural gain. The functions
below are plain `test_*` functions with bare asserts, so pytest WOULD
collect and run them unchanged if it is ever added -- this satisfies
both conventions rather than picking one.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app.services.reality_gap.parsers import (  # noqa: E402
    DATAFLASH_HEAD,
    EmptyLogError,
    LogMeta,
    LogParseError,
    ParsedLog,
    Signal,
    ULOG_MAGIC,
    UnsupportedLogFormat,
    _store,
    detect_format,
    parse_log,
    unwrap_yaw,
)
from app.services.reality_gap.align import (  # noqa: E402
    CANONICAL_RATE_HZ,
    FINE_SEARCH_S,
    AlignmentError,
    airborne_window,
    align_logs,
    classify_phases,
    find_arm_anchor,
    resample_pair,
    resample_to_grid,
)
from tests.fixtures.reality_gap import make_fixtures  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "reality_gap")
REAL_ULG = os.path.join(FIXTURES, "synthetic_real.ulg")
SIM_ULG = os.path.join(FIXTURES, "synthetic_sim.ulg")
REAL_BIN = os.path.join(FIXTURES, "synthetic_real.bin")
TRUNCATED = os.path.join(FIXTURES, "truncated.ulg")
EMPTY = os.path.join(FIXTURES, "empty.ulg")
GARBAGE = os.path.join(FIXTURES, "garbage.bin")


def _require_fixtures():
    missing = [p for p in (REAL_ULG, SIM_ULG, REAL_BIN, TRUNCATED, EMPTY, GARBAGE) if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            "Missing fixtures: "
            + ", ".join(os.path.basename(m) for m in missing)
            + "\nRun: python3 tests/fixtures/reality_gap/make_fixtures.py"
        )


# ===========================================================================
# Format detection (spec 2.1)
# ===========================================================================


def test_detect_format_by_magic_bytes():
    assert detect_format(REAL_ULG) == "ulog"
    assert detect_format(SIM_ULG) == "ulog"
    assert detect_format(REAL_BIN) == "dataflash"
    print("PASS: detect_format identifies ULog and DataFlash by magic bytes")


def test_magic_bytes_match_the_spec_values():
    """Spec 2.1 states the ULog magic as 55 4C 6F 67 01 12 35 and the
    DataFlash header as A3 95. Verified against the actual fixture bytes
    rather than trusting the constants in our own source."""
    with open(REAL_ULG, "rb") as handle:
        assert handle.read(7) == ULOG_MAGIC == b"\x55\x4c\x6f\x67\x01\x12\x35"
    with open(REAL_BIN, "rb") as handle:
        assert handle.read(2) == DATAFLASH_HEAD == b"\xa3\x95"
    print("PASS: on-disk magic bytes match the spec 2.1 values exactly")


def test_magic_beats_extension(tmp_name="renamed_ulog.bin"):
    """Spec 2.1 [DECISION]: 'format detection is by magic bytes first,
    extension second -- never extension alone.' A ULog saved as .bin must
    still be detected as a ULog; users rename these constantly."""
    path = os.path.join(FIXTURES, tmp_name)
    with open(REAL_ULG, "rb") as src, open(path, "wb") as dst:
        dst.write(src.read())
    try:
        assert detect_format(path) == "ulog", "extension won over magic bytes"
    finally:
        os.remove(path)
    print("PASS: magic bytes take priority over a misleading file extension")


def test_unsupported_format_rejected():
    """Spec 9 case 17."""
    try:
        detect_format(GARBAGE)
    except UnsupportedLogFormat as exc:
        assert ".ulg" in str(exc) and ".bin" in str(exc), "error should list accepted extensions"
        print("PASS: unrecognised file raises UnsupportedLogFormat listing accepted formats")
        return
    raise AssertionError("garbage file was not rejected")


def test_tlog_is_a_named_gap_not_a_silent_one():
    """Pass-1 scope excludes .tlog. It must fail loudly and say why --
    never return an empty result that looks like a parsed-but-empty log."""
    path = os.path.join(FIXTURES, "placeholder.tlog")
    with open(path, "wb") as handle:
        handle.write(b"\xfd" + b"\x00" * 64)
    try:
        parse_log(path)
    except UnsupportedLogFormat as exc:
        assert "tlog" in str(exc).lower()
        print("PASS: .tlog raises an explicit UnsupportedLogFormat naming it as a tracked gap")
        return
    finally:
        os.remove(path)
    raise AssertionError(".tlog did not raise")


# ===========================================================================
# ULog parsing (spec 5.2)
# ===========================================================================


def test_ulog_parses_to_canonical_schema():
    parsed = parse_log(REAL_ULG)
    expected = {
        "att.roll", "att.pitch", "att.yaw",
        "gyro.x", "gyro.y", "gyro.z",
        "accel.x", "accel.y", "accel.z",
        "pos.x", "pos.y", "pos.z",
        "vel.x", "vel.y", "vel.z",
        "alt.rel", "gps.fix_type", "gps.nsats", "gps.eph",
        "act.out0", "act.out1", "act.out2", "act.out3",
        "batt.voltage", "batt.current", "cpu.load", "mem.free",
    }
    missing = expected - set(parsed.signals)
    assert not missing, f"canonical keys missing from ULog parse: {sorted(missing)}"
    assert all(s.values.dtype == np.float32 for s in parsed.signals.values()), \
        "spec 5.7 requires float32 storage"
    print(f"PASS: ULog -> canonical schema, {len(parsed.signals)} signals, all float32")


def test_ulog_quaternion_to_euler_recovers_known_attitude():
    """The fixture's attitude was GENERATED from known Euler angles and
    written as a quaternion. Parsing must recover the original angles --
    this is what proves the q->Euler conversion in spec 5.2, not just
    that some plausible numbers came out."""
    parsed = parse_log(REAL_ULG)
    t = make_fixtures._t(make_fixtures.ATT_RATE_HZ)
    roll_expected, pitch_expected, _ = make_fixtures._mission(t)

    roll = parsed.signals["att.roll"].values
    pitch = parsed.signals["att.pitch"].values
    n = min(roll.size, roll_expected.size)
    max_roll_err = float(np.max(np.abs(roll[:n] - roll_expected[:n])))
    max_pitch_err = float(np.max(np.abs(pitch[:n] - pitch_expected[:n])))
    assert max_roll_err < 1e-4, f"roll error {max_roll_err}"
    assert max_pitch_err < 1e-4, f"pitch error {max_pitch_err}"
    print(f"PASS: quaternion->Euler recovers injected roll/pitch "
          f"(max err {max_roll_err:.2e} / {max_pitch_err:.2e} rad)")


def test_yaw_is_unwrapped():
    """Spec 9 case 11. The fixture's yaw deliberately sweeps past +/-pi.
    After unwrapping there must be no 2*pi discontinuity, and the range
    must exceed +/-pi (which a wrapped signal can never do)."""
    parsed = parse_log(REAL_ULG)
    yaw = parsed.signals["att.yaw"].values.astype(np.float64)
    jumps = np.abs(np.diff(yaw))
    assert jumps.max() < math.pi, f"found a {jumps.max():.2f} rad jump -- yaw was not unwrapped"
    assert yaw.max() > math.pi or yaw.min() < -math.pi, \
        "fixture yaw never crossed +/-pi, so this test proves nothing"
    print(f"PASS: yaw unwrapped (range {yaw.min():.2f}..{yaw.max():.2f} rad, max step {jumps.max():.4f})")


def test_unwrap_yaw_on_synthetic_sweep():
    """Direct unit test of the helper on a -pi -> +pi -> -pi sweep
    (spec 10.1 item 8's setup), independent of any fixture."""
    t = np.linspace(0, 4 * np.pi, 2000)
    wrapped = np.arctan2(np.sin(t), np.cos(t)).astype(np.float32)
    unwrapped = unwrap_yaw(wrapped).astype(np.float64)
    assert np.abs(np.diff(unwrapped)).max() < math.pi
    # A monotonically increasing angle must unwrap to a monotonic ramp.
    assert unwrapped[-1] > unwrapped[0] + 3 * math.pi
    print("PASS: unwrap_yaw turns a wrapped +/-pi sweep into a monotonic ramp")


def test_actuator_outputs_normalised_to_unit_range():
    """Spec 5.2 [DECISION]: actuators normalised to [0,1] using the log's
    own PWM_MIN/PWM_MAX. The fixture writes 1300-1700 us against a
    declared 1000/2000 range, so the exact expected result is 0.3-0.7."""
    parsed = parse_log(REAL_ULG)
    out0 = parsed.signals["act.out0"].values
    assert 0.0 <= out0.min() <= 1.0 and 0.0 <= out0.max() <= 1.0
    assert abs(float(out0.min()) - 0.30) < 1e-3, f"min {out0.min()}"
    assert abs(float(out0.max()) - 0.70) < 1e-3, f"max {out0.max()}"
    assert parsed.meta.actuator_pwm_range == (1000.0, 2000.0)
    print(f"PASS: actuator PWM 1300-1700us normalised to {out0.min():.3f}-{out0.max():.3f}")


def _build_ulog_with_n_motors(path, n_motors, *, with_rotor_count_param):
    """
    A minimal, real ULog (round-tripped through pyulog's own writer, same
    convention as make_fixtures.py) whose actuator_outputs message has
    the FIXED 16-slot struct PX4 actually logs -- motors 0..n_motors-1
    carry a real PWM signal, the rest sit at exactly 0.0, matching what
    was verified on the real reference fixture (px4_641d242a.ulg).

    `with_rotor_count_param` toggles whether CA_ROTOR_COUNT is present,
    so both count sources (parameter vs. nonzero-heuristic fallback) get
    a real fixture to run against, not just a mocked dict.
    """
    from pyulog import ULog

    n = 2000
    t_us = (np.arange(n, dtype=np.float64) * 20000).astype(np.uint64)  # 50 Hz

    ulog = ULog(None)
    ulog._file_version = 1
    ulog._start_timestamp = 0
    ulog._msg_info_dict = {"ver_sw": "v1.14.0-synthetic"}
    ulog._msg_info_dict_types = {"ver_sw": "char[16]"}
    params = {"SYS_AUTOSTART": 4001.0, "PWM_MAIN_MIN": 1000.0, "PWM_MAIN_MAX": 2000.0}
    if with_rotor_count_param:
        params["CA_ROTOR_COUNT"] = float(n_motors)
    ulog._initial_parameters = params

    fields = [("uint64_t", "timestamp")] + [("float", f"output[{i}]") for i in range(16)]
    data = {"timestamp": t_us}
    for i in range(16):
        if i < n_motors:
            # A real, distinct PWM trace per motor -- never exactly 0.0,
            # which is what distinguishes a driven motor from padding.
            data[f"output[{i}]"] = (1500.0 + 50.0 * (i + 1)
                                    + 20.0 * np.sin(np.arange(n) / 30.0)).astype(np.float32)
        else:
            data[f"output[{i}]"] = np.zeros(n, dtype=np.float32)  # unused struct slot

    ulog._message_formats["actuator_outputs"] = make_fixtures._message_format(
        "actuator_outputs", fields)
    ulog._data_list = [make_fixtures._dataset(0, "actuator_outputs", fields, data)]
    ulog.write_ulog(path)


def test_hexarotor_actuator_outputs_all_captured_and_warned():
    """Health spec amendment A-1 replaced spec 5.2's act.out[0..3] cap: every
    motor is captured, by motor number. This synthetic hex has CA_ROTOR_COUNT
    but no output-function parameters, so the channel -> motor mapping is an
    assumption, and the parser has to say so rather than present it as fact."""
    path = os.path.join(FIXTURES, "hexarotor_6motor.ulg")
    _build_ulog_with_n_motors(path, 6, with_rotor_count_param=True)
    try:
        parsed = parse_log(path)
        captured = sorted(k for k in parsed.signals if k.startswith("act.out"))
        assert captured == [f"act.out{i}" for i in range(6)], captured
        assert parsed.motor_info["motor_count"] == 6
        assert parsed.motor_info["motor_count_source"] == "CA_ROTOR_COUNT"
        assert parsed.motor_info["mapping_source"] == "channel_order_assumed"
        assert any(w.startswith("actuator_mapping_assumed:") for w in parsed.warnings), parsed.warnings
        assert not any(w.startswith("actuator_outputs_truncated:") for w in parsed.warnings)
        print("PASS: 6-motor log -> act.out0-5 all captured (A-1), mapping flagged as assumed")
    finally:
        os.remove(path)


def test_hexarotor_without_rotor_count_param_uses_nonzero_heuristic():
    """Older firmware has no CA_ROTOR_COUNT. The fallback must still find all
    six motors, distinguishing a real (always-nonzero) PWM trace from the
    always-exactly-0.0 padding in the unused struct slots -- verified against
    the real reference log's own padding behaviour."""
    path = os.path.join(FIXTURES, "hexarotor_6motor_noparam.ulg")
    _build_ulog_with_n_motors(path, 6, with_rotor_count_param=False)
    try:
        parsed = parse_log(path)
        assert parsed.motor_info["motor_count"] == 6, parsed.motor_info
        assert parsed.motor_info["motor_count_source"] == "nonzero_output_heuristic"
        assert sorted(k for k in parsed.signals if k.startswith("act.out")) == [f"act.out{i}" for i in range(6)]
        print("PASS: no CA_ROTOR_COUNT -> nonzero-value fallback still finds and captures 6 motors")
    finally:
        os.remove(path)


def test_quadrotor_actuator_outputs_no_false_positive_warning():
    """The other half of the correctness bar: a real quadrotor (4 driven
    outputs, 12 padding slots at 0.0) must NOT get a spurious truncation
    warning. Regression-checked directly against the real fixture too
    (px4_641d242a.ulg, CA_ROTOR_COUNT=4) in the assertions below."""
    path = os.path.join(FIXTURES, "quadrotor_4motor.ulg")
    _build_ulog_with_n_motors(path, 4, with_rotor_count_param=True)
    try:
        parsed = parse_log(path)
        assert not any(w.startswith("actuator_outputs_truncated:") for w in parsed.warnings)
    finally:
        os.remove(path)

    real = parse_log(REAL_ULG_REAL_FIXTURE := os.path.join(
        os.path.dirname(FIXTURES), "reality_gap", "real", "px4_641d242a.ulg"))
    if os.path.exists(REAL_ULG_REAL_FIXTURE):
        assert not any(w.startswith("actuator_outputs_truncated:") for w in real.warnings)
        print("PASS: 4-motor synthetic AND the real px4_641d242a.ulg quadrotor "
              "both produce zero false-positive truncation warnings")
    else:
        print("PASS: 4-motor synthetic produces zero false-positive truncation warning "
              "(real fixture not present in this checkout, skipped)")


def test_alt_rel_is_negated_ned_z():
    """Spec 5.2: alt.rel = vehicle_local_position.z * -1, because NED z is
    positive-DOWN. A sign error here would invert every altitude in the
    module, so it is asserted explicitly."""
    parsed = parse_log(REAL_ULG)
    alt = parsed.signals["alt.rel"].values
    pos_z = parsed.signals["pos.z"].values
    assert np.allclose(alt, -pos_z, atol=1e-5)
    assert alt.max() > 25.0, "fixture should climb to ~30 m"
    print(f"PASS: alt.rel is negated NED z (peak {alt.max():.1f} m)")


def test_native_rates_detected():
    """Spec 9 case 4 requires both native rates on every signal row."""
    parsed = parse_log(REAL_ULG)
    assert abs(parsed.signals["att.roll"].native_rate_hz - 50.0) < 0.5
    assert abs(parsed.signals["gyro.x"].native_rate_hz - 100.0) < 0.5
    assert abs(parsed.signals["gps.eph"].native_rate_hz - 2.0) < 0.1
    print("PASS: native rates detected (att 50 Hz, gyro 100 Hz, gps 2 Hz)")


def test_simulator_provenance_detection():
    """Spec 2.2: the sim fixture carries SIM_* params, a simulation-range
    SYS_AUTOSTART and a 'PX4 SITL' sys_name; the real fixture carries
    none of those and must come back INCONCLUSIVE (None), not False --
    absence of evidence is not evidence of a real flight."""
    real = parse_log(REAL_ULG)
    sim = parse_log(SIM_ULG)
    assert sim.meta.looks_simulated is True, "sim log not detected as simulated"
    assert len(sim.meta.sim_evidence) >= 2, sim.meta.sim_evidence
    assert real.meta.looks_simulated is None, \
        "real log should be inconclusive (None), never a confident False"
    print(f"PASS: sim provenance detected ({len(sim.meta.sim_evidence)} pieces of evidence); "
          f"real log correctly inconclusive")


# ===========================================================================
# Error handling (spec 9)
# ===========================================================================


def test_empty_log_raises_empty_log_error():
    """Spec 9 case 6: parses fine, but no mapped topics -> distinct error."""
    try:
        parse_log(EMPTY)
    except EmptyLogError as exc:
        assert "no usable telemetry" in str(exc)
        print("PASS: structurally-valid log with no mapped topics raises EmptyLogError")
        return
    raise AssertionError("empty log did not raise EmptyLogError")


def test_truncated_log_recovers_partial_data():
    """Spec 9 case 7: 'ULog files truncated mid-flight are extremely
    common -- pyulog recovers partial data, so attempt partial parse
    first and only fail if the recovered data is empty.' The truncated
    fixture must therefore PARSE, not raise."""
    parsed = parse_log(TRUNCATED)
    assert len(parsed.signals) > 0, "no data recovered from truncated log"
    full = parse_log(REAL_ULG)
    truncated_samples = parsed.signals["att.roll"].t_us.size
    full_samples = full.signals["att.roll"].t_us.size
    assert truncated_samples < full_samples, "truncated log should have fewer samples"
    print(f"PASS: truncated log recovered {len(parsed.signals)} signals, "
          f"{truncated_samples}/{full_samples} attitude samples (no exception)")


def test_parse_errors_are_typed_not_bare_exceptions():
    """Spec 9 case 7 requires 'never a 500' -- every failure path must be
    a LogParseError subclass the endpoint layer can turn into a 422/415."""
    for path in (GARBAGE,):
        try:
            parse_log(path)
        except LogParseError:
            pass
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"{path} raised a non-LogParseError: {type(exc).__name__}") from exc
    print("PASS: parse failures raise LogParseError subclasses, never bare exceptions")


def test_nan_and_inf_samples_are_dropped():
    """Spec 9 case 12: drop the bad samples from THAT signal only and
    record the count -- never drop the whole row, never zero-fill."""
    signals, warnings = {}, []
    t = np.arange(100, dtype=np.int64) * 20_000
    values = np.ones(100, dtype=np.float64)
    values[10] = np.nan
    values[20] = np.inf
    values[30] = -np.inf
    _store(signals, warnings, "test.sig", t, values)
    sig = signals["test.sig"]
    assert sig.values.size == 97, f"expected 97 surviving samples, got {sig.values.size}"
    assert sig.dropped_samples == 3
    assert np.all(np.isfinite(sig.values))
    print("PASS: NaN/Inf samples dropped per-signal with dropped_samples recorded")


def test_clock_jump_truncates_to_longer_fragment():
    """Spec 9 case 13: on a backwards jump > 1 s, cut and keep the longer
    fragment, with a clock_jump_truncated warning."""
    signals, warnings = {}, []
    t = np.concatenate([
        np.arange(30, dtype=np.int64) * 20_000,                      # 0.0 - 0.6 s
        np.arange(200, dtype=np.int64) * 20_000 + 100_000_000,       # jumps FORWARD then
    ])
    t[30:] -= 150_000_000  # ...pull the tail backwards by 1.5 s -> a real backwards jump
    values = np.arange(t.size, dtype=np.float64)
    _store(signals, warnings, "test.sig", t, values)
    assert "clock_jump_truncated" in warnings, warnings
    sig = signals["test.sig"]
    assert sig.t_us.size == 200, f"should keep the 200-sample fragment, kept {sig.t_us.size}"
    assert np.all(np.diff(sig.t_us) > 0), "result must be strictly increasing"
    print("PASS: backwards clock jump truncated to the longer fragment, warning raised")


def test_duplicate_timestamps_deduplicated():
    signals, warnings = {}, []
    t = np.array([0, 20_000, 20_000, 40_000, 40_000, 60_000], dtype=np.int64)
    values = np.array([1.0, 2.0, 99.0, 3.0, 99.0, 4.0])
    _store(signals, warnings, "test.sig", t, values)
    sig = signals["test.sig"]
    assert sig.t_us.size == 4, f"expected 4 unique timestamps, got {sig.t_us.size}"
    assert np.all(np.diff(sig.t_us) > 0)
    print("PASS: duplicate timestamps deduplicated keeping the first")


# ===========================================================================
# DataFlash parsing (spec 5.2)
# ===========================================================================


def test_dataflash_parses_to_same_canonical_schema():
    parsed = parse_log(REAL_BIN)
    for key in ("att.roll", "att.pitch", "att.yaw", "gyro.x", "accel.z",
                "alt.rel", "act.out0", "batt.voltage", "vibe.x"):
        assert key in parsed.signals, f"{key} missing from DataFlash parse"
    assert all(s.values.dtype == np.float32 for s in parsed.signals.values())
    print(f"PASS: DataFlash -> same canonical schema, {len(parsed.signals)} signals")


def test_dataflash_degrees_converted_to_radians():
    """ArduPilot logs attitude in DEGREES; the canonical schema is
    radians. The fixture writes the same underlying mission as the ULog,
    so the two parsers must agree -- this cross-checks the unit
    conversion against an independent implementation path rather than
    against a hardcoded number."""
    ulog = parse_log(REAL_ULG)
    dflash = parse_log(REAL_BIN)
    n = min(ulog.signals["att.roll"].values.size, dflash.signals["att.roll"].values.size)
    err = float(np.max(np.abs(
        ulog.signals["att.roll"].values[:n] - dflash.signals["att.roll"].values[:n]
    )))
    assert err < 1e-3, f"ULog and DataFlash attitude disagree by {err} rad"
    peak = float(np.max(np.abs(dflash.signals["att.roll"].values)))
    assert peak < math.pi, f"peak {peak} looks like degrees, not radians"
    print(f"PASS: DataFlash degrees->radians agrees with ULog path to {err:.2e} rad")


def test_hdop_is_not_conflated_with_eph():
    """Spec 5.2 [DECISION]: 'gps.eph vs GPS.HDop are not the same
    quantity ... multiplying HDop by an assumed UERE to fake metres would
    be exactly the fabricated precision this project forbids.' The
    DataFlash parser must therefore expose HDop under its own key and
    must NOT populate gps.eph."""
    dflash = parse_log(REAL_BIN)
    assert "gps.hdop" in dflash.signals, "HDop should be preserved under its own key"
    assert "gps.eph" not in dflash.signals, \
        "DataFlash HDop was wrongly mapped to gps.eph -- these are different quantities"
    ulog = parse_log(REAL_ULG)
    assert "gps.eph" in ulog.signals and "gps.hdop" not in ulog.signals
    print("PASS: HDop and eph kept distinct (no fabricated unit conversion)")


# ===========================================================================
# Alignment (spec 5.3)
# ===========================================================================


def _synthetic_parsed(shift_s=0.0, duration_s=60.0, rate_hz=50.0, seed=1, phase=0.0,
                      with_takeoff=False, name="synthetic"):
    """Build a ParsedLog in memory with a controlled time shift.

    WHY IN-MEMORY RATHER THAN VIA A FIXTURE FILE: the file fixtures'
    coarse anchor (takeoff detection) legitimately absorbs a whole-mission
    shift, which is correct behaviour but means it never exercises the
    FINE cross-correlation stage. These synthetic logs deliberately omit
    a takeoff signature so the coarse anchor falls back to first-sample
    and leaves a known residual shift for the fine stage to recover.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_s, 1.0 / rate_hz)
    roll = (0.3 * np.sin(2 * np.pi * 0.11 * (t + shift_s) + phase)
            + 0.1 * np.sin(2 * np.pi * 0.43 * (t + shift_s) + phase)
            + rng.standard_normal(t.size) * 0.002)
    t_us = (t * 1e6).astype(np.int64)
    signals = {
        "att.roll": Signal("att.roll", t_us, roll.astype(np.float32)),
        "alt.rel": Signal("alt.rel", t_us, np.full(t.size, 10.0, dtype=np.float32)),
    }
    if with_takeoff:
        az = np.full(t.size, -9.81, dtype=np.float32)
        az[(t >= 5.0) & (t < 7.0)] -= 2.5
        signals["accel.z"] = Signal("accel.z", t_us, az)
    meta = LogMeta(source_format="ulog", filename=f"{name}.ulg", sha256="0" * 64)
    return ParsedLog(signals=signals, params={}, meta=meta)


def test_fine_alignment_recovers_known_shift():
    """Spec 5.3 step 2. A known 0.60 s residual shift must be recovered
    by the band-passed cross-correlation to within one grid sample
    (1/50 Hz = 20 ms)."""
    known_shift_s = 0.60
    real = _synthetic_parsed(shift_s=0.0, seed=1, name="real")
    sim = _synthetic_parsed(shift_s=known_shift_s, seed=2, name="sim")
    result = align_logs(real, sim)
    assert result.quality == "fine", f"expected fine alignment, got {result.quality}"
    error_s = abs(abs(result.fine_shift_s) - known_shift_s)
    assert error_s <= 1.0 / CANONICAL_RATE_HZ + 1e-9, \
        f"recovered {result.fine_shift_s:.3f}s vs known {known_shift_s}s (err {error_s:.4f}s)"
    assert result.fine_correlation >= 0.5
    print(f"PASS: fine alignment recovered a {known_shift_s}s shift as "
          f"{abs(result.fine_shift_s):.3f}s (rho={result.fine_correlation:.3f})")


def test_decimated_50hz_matches_full_rate_500hz_within_one_grid_sample():
    """The fine aligner correlates on the 50 Hz canonical grid, not on a
    native 500 Hz att.roll. That decimation is only legitimate if it
    recovers the same lag: the 0.2-5 Hz band-pass means anything above
    ~12.5 Hz is redundant for this measurement. Asserted against a
    500 Hz reference of the same signal with a known injected shift."""
    from app.services.reality_gap.align import (
        _bandpass, _highest_variance_window, _normalised_cross_correlation,
    )

    known_shift_s = 0.614
    for rate, win_s in ((500.0, 120.0), (CANONICAL_RATE_HZ, 120.0)):
        n = int(rate * 300)
        t = np.arange(n) / rate
        mission = (0.30 * np.sin(2 * np.pi * 0.11 * t)
                   + 0.10 * np.sin(2 * np.pi * 0.43 * t)
                   + 0.05 * np.sin(2 * np.pi * 1.7 * t))
        shifted = (0.30 * np.sin(2 * np.pi * 0.11 * (t - known_shift_s))
                   + 0.10 * np.sin(2 * np.pi * 0.43 * (t - known_shift_s))
                   + 0.05 * np.sin(2 * np.pi * 1.7 * (t - known_shift_s)))
        a, b = _bandpass(mission, rate), _bandpass(shifted, rate)
        w = _highest_variance_window(a, int(win_s * rate))
        lag, rho = _normalised_cross_correlation(a[w], b[w], int(FINE_SEARCH_S * rate))
        recovered = abs(lag / rate)
        if rate == 500.0:
            full_rate_shift = recovered
        else:
            grid = 1.0 / CANONICAL_RATE_HZ
            assert abs(recovered - full_rate_shift) <= grid, (
                f"50 Hz recovered {recovered:.4f}s vs 500 Hz {full_rate_shift:.4f}s "
                f"-- differ by more than one {grid*1000:.0f} ms grid sample")
            assert rho >= 0.5
            print(f"PASS: 50 Hz recovered {recovered:.4f}s vs 500 Hz "
                  f"{full_rate_shift:.4f}s (known {known_shift_s}s), "
                  f"within one {grid*1000:.0f} ms grid sample")


def test_variance_window_beats_from_the_start():
    """A flight that is quiescent at the start and manoeuvres later: taking
    the first N samples finds nothing to correlate, the variance-selected
    window finds the manoeuvre."""
    from app.services.reality_gap.align import (
        _bandpass, _highest_variance_window, _normalised_cross_correlation,
    )
    rate, shift_s = CANONICAL_RATE_HZ, 0.60
    n = int(rate * 400)
    t = np.arange(n) / rate
    active = (t > 250).astype(float)          # nothing happens for the first 250 s
    m = active * (0.3 * np.sin(2 * np.pi * 0.11 * t) + 0.1 * np.sin(2 * np.pi * 0.43 * t))
    a_full = _bandpass(m, rate)
    active_s = (t > 250).astype(float)
    ms = active_s * (0.3 * np.sin(2 * np.pi * 0.11 * (t - shift_s))
                     + 0.1 * np.sin(2 * np.pi * 0.43 * (t - shift_s)))
    b_full = _bandpass(ms, rate)

    win = int(120.0 * rate)
    _, rho_start = _normalised_cross_correlation(
        a_full[:win], b_full[:win], int(FINE_SEARCH_S * rate))
    w = _highest_variance_window(a_full, win)
    lag_var, rho_var = _normalised_cross_correlation(
        a_full[w], b_full[w], int(FINE_SEARCH_S * rate))

    assert rho_var > rho_start, f"variance window rho={rho_var} not better than start rho={rho_start}"
    assert rho_var >= 0.5, f"variance window still below the gate: {rho_var}"
    assert abs(abs(lag_var / rate) - shift_s) <= 1.0 / rate
    print(f"PASS: variance window rho={rho_var:.3f} vs from-the-start rho={rho_start:.3f}; "
          f"recovered {abs(lag_var/rate):.2f}s")


def test_unrelated_flights_fall_back_to_coarse_only():
    """Spec 5.3 [DECISION] fine-alignment gate -- 'the single most
    important honesty guard in the module'. Two unrelated missions must
    NOT get a confident point-wise alignment."""
    real = _synthetic_parsed(seed=1, name="real")
    rng = np.random.default_rng(99)
    t = np.arange(0.0, 60.0, 1.0 / 50.0)
    unrelated = rng.standard_normal(t.size) * 0.3  # pure noise: no shared structure
    sim = _synthetic_parsed(seed=2, name="sim")
    sim.signals["att.roll"] = Signal("att.roll", (t * 1e6).astype(np.int64),
                                     unrelated.astype(np.float32))
    result = align_logs(real, sim)
    assert result.quality == "coarse_only", \
        f"unrelated flights were confidently aligned (rho={result.fine_correlation})"
    assert "fine_alignment_below_gate" in result.warnings
    assert result.fine_shift_s == 0.0, "no shift may be applied when the gate fails"
    print(f"PASS: unrelated flights -> coarse_only (rho={result.fine_correlation:.3f}), no shift applied")


def test_insufficient_overlap_raises():
    """Spec 9 case 1: overlap < 20 s -> refuse with the actual number."""
    real = _synthetic_parsed(duration_s=60.0, seed=1, name="real")
    sim = _synthetic_parsed(duration_s=12.0, seed=2, name="sim")
    try:
        align_logs(real, sim)
    except AlignmentError as exc:
        assert "overlap" in str(exc).lower()
        print(f"PASS: insufficient overlap refused -- {exc}")
        return
    raise AssertionError("short overlap was not rejected")


def test_takeoff_anchor_detected():
    """Spec 5.3 step 1 fallback: no arm event -> takeoff detection at the
    point |accel.z| departs 1 g by >1.0 m/s^2 sustained 0.5 s."""
    parsed = _synthetic_parsed(with_takeoff=True, name="real")
    anchor_us, method = find_arm_anchor(parsed)
    assert method == "takeoff_detect", f"got {method}"
    anchor_s = anchor_us / 1e6
    assert 4.9 <= anchor_s <= 5.6, f"takeoff detected at {anchor_s:.2f}s, expected ~5.0s"
    print(f"PASS: takeoff anchor detected at {anchor_s:.2f}s (injected at 5.00s)")


def test_first_sample_fallback_warns():
    """Spec 5.3: neither arm event nor takeoff -> first sample, WITH a
    warning so the report never silently claims a real anchor."""
    real = _synthetic_parsed(seed=1, name="real")   # no accel.z at all
    sim = _synthetic_parsed(seed=2, name="sim")
    _, method = find_arm_anchor(real)
    assert method == "first_sample"
    result = align_logs(real, sim)
    assert "coarse_alignment_fallback_first_sample" in result.warnings
    print("PASS: first-sample anchor fallback records coarse_alignment_fallback_first_sample")


def test_arm_event_preferred_over_takeoff():
    """The arm-event branch is real code, not dead code -- it fires when
    a status.armed signal is present."""
    parsed = _synthetic_parsed(with_takeoff=True, name="real")
    t_us = parsed.signals["att.roll"].t_us
    armed = np.zeros(t_us.size, dtype=np.float32)
    armed[t_us >= 2_000_000] = 1.0
    parsed.signals["status.armed"] = Signal("status.armed", t_us, armed)
    anchor_us, method = find_arm_anchor(parsed)
    assert method == "arm_event", f"got {method}"
    assert abs(anchor_us / 1e6 - 2.0) < 0.05
    print("PASS: an explicit arm event takes priority over takeoff detection")


# ===========================================================================
# Resampling (spec 5.4)
# ===========================================================================


def test_resample_produces_canonical_50hz_grid():
    real = parse_log(REAL_ULG)
    sim = parse_log(SIM_ULG)
    pairs = resample_pair(real, sim, align_logs(real, sim))
    pair = pairs["att.roll"]
    dt = np.diff(pair.t_grid_s)
    assert abs(float(np.median(dt)) - 1.0 / CANONICAL_RATE_HZ) < 1e-6, \
        f"grid spacing {np.median(dt)} != 1/50 s"
    assert pair.real_grid.size == pair.sim_grid.size == pair.t_grid_s.size
    print(f"PASS: resampled onto the canonical {CANONICAL_RATE_HZ:.0f} Hz grid "
          f"({pair.real_grid.size} samples)")


def test_zero_order_hold_for_enum_signals():
    """Spec 5.4: 'interpolating a fix type produces fix type 2.7, which is
    nonsense.' Every resampled fix-type value must be an exact integer."""
    sig = Signal("gps.fix_type",
                 np.array([0, 1_000_000, 2_000_000, 3_000_000], dtype=np.int64),
                 np.array([2.0, 3.0, 3.0, 4.0], dtype=np.float32))
    _, values = resample_to_grid(sig, 0, 3_000_000, 50.0, zero_order_hold=True)
    finite = values[np.isfinite(values)]
    assert np.all(finite == np.round(finite)), "zero-order hold produced fractional enum values"
    assert set(np.unique(finite)).issubset({2.0, 3.0, 4.0})
    print("PASS: enum signals resampled by zero-order hold (no fractional fix types)")


def test_linear_interpolation_for_continuous_signals():
    sig = Signal("att.roll",
                 np.array([0, 1_000_000], dtype=np.int64),
                 np.array([0.0, 1.0], dtype=np.float32))
    _, values = resample_to_grid(sig, 0, 1_000_000, 50.0, zero_order_hold=False)
    midpoint = values[values.size // 2]
    assert abs(float(midpoint) - 0.5) < 0.02, f"midpoint {midpoint} not interpolated"
    print("PASS: continuous signals linearly interpolated")


def test_no_extrapolation_beyond_native_span():
    """A resample window wider than the signal's own coverage must yield
    NaN outside it, not a fabricated flat reading. np.interp would
    silently clamp to the endpoint value, inventing data for a period the
    sensor never covered."""
    sig = Signal("att.roll",
                 np.array([1_000_000, 2_000_000], dtype=np.int64),
                 np.array([1.0, 2.0], dtype=np.float32))
    grid_us, values = resample_to_grid(sig, 0, 3_000_000, 50.0, zero_order_hold=False)
    assert np.isnan(values[0]), "extrapolated before the first sample"
    assert np.isnan(values[-1]), "extrapolated past the last sample"
    assert np.any(np.isfinite(values)), "everything was NaN -- window/logic wrong"
    print("PASS: no extrapolation outside a signal's measured span (NaN instead)")


def test_low_rate_signals_flagged():
    """Spec 9 case 5: a 2 Hz GPS signal is below the 5 Hz gate."""
    real = parse_log(REAL_ULG)
    sim = parse_log(SIM_ULG)
    pairs = resample_pair(real, sim, align_logs(real, sim))
    assert "low_rate" in pairs["gps.eph"].flags, pairs["gps.eph"].flags
    assert "low_rate" not in pairs["att.roll"].flags
    assert "upsampled_to_grid" in pairs["gps.eph"].flags
    print("PASS: 2 Hz GPS flagged low_rate + upsampled_to_grid; 50 Hz attitude unflagged")


def test_native_samples_preserved_alongside_grid():
    """Spec 5.4: distribution/spectral metrics must run on NATIVE samples,
    so both representations have to survive resampling."""
    real = parse_log(REAL_ULG)
    sim = parse_log(SIM_ULG)
    pairs = resample_pair(real, sim, align_logs(real, sim))
    gps = pairs["gps.eph"]
    assert gps.real_native.size > 0
    assert gps.real_native.size < gps.real_grid.size, \
        "native 2 Hz samples should be far fewer than the 50 Hz grid"
    print(f"PASS: native samples preserved ({gps.real_native.size} native vs "
          f"{gps.real_grid.size} grid for 2 Hz GPS)")


# ===========================================================================
# Segment selection (spec 5.5)
# ===========================================================================


def test_airborne_window_trims_ground_time():
    """Spec 5.5: ground time 'would flatter the simulator', so metrics run
    from the point altitude exceeds 0.5 m."""
    parsed = parse_log(REAL_ULG)
    anchor_us, _ = find_arm_anchor(parsed)
    start_us, end_us, warnings = airborne_window(parsed, anchor_us)
    alt = parsed.signals["alt.rel"]
    inside = (alt.t_us >= start_us) & (alt.t_us <= end_us)
    assert float(alt.values[inside].max()) > 25.0
    assert "ground_only" not in warnings
    duration = (end_us - start_us) / 1e6
    assert 40.0 < duration < 55.0, f"airborne window {duration:.1f}s looks wrong"
    print(f"PASS: airborne window is {duration:.1f}s of the 60s log (ground time trimmed)")


def test_ground_only_flight_warns_but_does_not_fail():
    """Spec 9 case 9: armed but never above 0.5 m -> proceed with warning."""
    parsed = _synthetic_parsed(name="real")
    parsed.signals["alt.rel"] = Signal(
        "alt.rel", parsed.signals["att.roll"].t_us,
        np.full(parsed.signals["att.roll"].t_us.size, 0.1, dtype=np.float32))
    _, _, warnings = airborne_window(parsed, 0)
    assert "ground_only" in warnings
    print("PASS: never-airborne log warns ground_only rather than failing")


def test_phase_classification():
    """Spec 5.5 [DECISION]: climb vz < -0.5, cruise |vz| <= 0.5,
    descent vz > 0.5 (NED, negative up). A phase with < 250 samples is
    omitted entirely rather than reported from too little data."""
    vz = np.concatenate([
        np.full(400, -2.0),   # climb
        np.full(600, 0.0),    # cruise
        np.full(400, 2.0),    # descent
        np.full(50, -3.0),    # too short -> must not create a second climb entry
    ])
    phases = classify_phases(vz)
    assert set(phases) == {"climb", "cruise", "descent"}, sorted(phases)
    assert int(phases["climb"].sum()) == 450   # 400 + the 50 short ones, same mask
    assert int(phases["cruise"].sum()) == 600
    print("PASS: phases classified by vertical velocity with the 250-sample minimum")


def test_short_phase_omitted():
    vz = np.concatenate([np.full(600, 0.0), np.full(10, -2.0)])
    phases = classify_phases(vz)
    assert "cruise" in phases
    assert "climb" not in phases, "a 10-sample climb should be omitted, not reported"
    print("PASS: a phase below the 250-sample minimum is omitted entirely")


# ===========================================================================
# Determinism (spec 10.3 item 16)
# ===========================================================================


def test_pipeline_is_deterministic():
    """Spec 10.3 item 16: 'the same log pair processed twice produces
    byte-identical metrics. No RNG anywhere in the pipeline without a
    fixed seed.'"""
    results = []
    for _ in range(2):
        real = parse_log(REAL_ULG)
        sim = parse_log(SIM_ULG)
        alignment = align_logs(real, sim)
        pairs = resample_pair(real, sim, alignment)
        results.append((alignment.fine_shift_s, alignment.fine_correlation,
                        pairs["att.roll"].real_grid.tobytes(),
                        pairs["att.roll"].sim_grid.tobytes()))
    assert results[0] == results[1], "pipeline output differs between identical runs"
    print("PASS: parse -> align -> resample is byte-identical across repeated runs")


def test_cross_format_comparison_works():
    """Spec 10.2 item 12: a .ulg real vs .bin sim comparison must complete.
    This is the payoff of normalising units at the parser boundary."""
    real = parse_log(REAL_ULG)
    sim = parse_log(REAL_BIN)
    alignment = align_logs(real, sim)
    pairs = resample_pair(real, sim, alignment)
    shared = set(pairs)
    assert "att.roll" in shared and "gyro.x" in shared and "alt.rel" in shared
    # gps.eph (ULog) and gps.hdop (DataFlash) must NOT pair up -- they are
    # the incommensurable case of spec 5.2.
    assert "gps.eph" not in shared and "gps.hdop" not in shared
    print(f"PASS: cross-format ULog-vs-DataFlash comparison produced {len(shared)} shared signals, "
          f"with eph/HDop correctly not paired")


ALL_TESTS = [
    test_detect_format_by_magic_bytes,
    test_magic_bytes_match_the_spec_values,
    test_magic_beats_extension,
    test_unsupported_format_rejected,
    test_tlog_is_a_named_gap_not_a_silent_one,
    test_ulog_parses_to_canonical_schema,
    test_ulog_quaternion_to_euler_recovers_known_attitude,
    test_yaw_is_unwrapped,
    test_unwrap_yaw_on_synthetic_sweep,
    test_actuator_outputs_normalised_to_unit_range,
    test_hexarotor_actuator_outputs_all_captured_and_warned,
    test_hexarotor_without_rotor_count_param_uses_nonzero_heuristic,
    test_quadrotor_actuator_outputs_no_false_positive_warning,
    test_alt_rel_is_negated_ned_z,
    test_native_rates_detected,
    test_simulator_provenance_detection,
    test_empty_log_raises_empty_log_error,
    test_truncated_log_recovers_partial_data,
    test_parse_errors_are_typed_not_bare_exceptions,
    test_nan_and_inf_samples_are_dropped,
    test_clock_jump_truncates_to_longer_fragment,
    test_duplicate_timestamps_deduplicated,
    test_dataflash_parses_to_same_canonical_schema,
    test_dataflash_degrees_converted_to_radians,
    test_hdop_is_not_conflated_with_eph,
    test_fine_alignment_recovers_known_shift,
    test_decimated_50hz_matches_full_rate_500hz_within_one_grid_sample,
    test_variance_window_beats_from_the_start,
    test_unrelated_flights_fall_back_to_coarse_only,
    test_insufficient_overlap_raises,
    test_takeoff_anchor_detected,
    test_first_sample_fallback_warns,
    test_arm_event_preferred_over_takeoff,
    test_resample_produces_canonical_50hz_grid,
    test_zero_order_hold_for_enum_signals,
    test_linear_interpolation_for_continuous_signals,
    test_no_extrapolation_beyond_native_span,
    test_low_rate_signals_flagged,
    test_native_samples_preserved_alongside_grid,
    test_airborne_window_trims_ground_time,
    test_ground_only_flight_warns_but_does_not_fail,
    test_phase_classification,
    test_short_phase_omitted,
    test_pipeline_is_deterministic,
    test_cross_format_comparison_works,
]



# ===========================================================================
# PASS 2A -- metrics (spec 3.1-3.9) and spectral (3.7-3.8), per spec 10.1
# ===========================================================================

from app.services.reality_gap.metrics import (  # noqa: E402
    compare_signal, effective_sample_size, freedman_diaconis_edges,
    kl_divergence_bits, resource_panel, _smoothed_histograms,
)
from app.services.reality_gap.spectral import (  # noqa: E402
    ACCEL_CHAIN, BANDS, actuator_lag, band_ratios_db, compare_spectra,
    estimate_lag_s, welch_nperseg, welch_psd,
)
from scipy import stats as _sp_stats  # noqa: E402


def _mission_signal(n=6000, seed=11, fs=50.0):
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)
    return (0.3 * np.sin(2 * np.pi * 0.11 * t) + 0.1 * np.sin(2 * np.pi * 0.43 * t)
            + rng.standard_normal(n) * 0.01)


# --- 10.1 item 1: synthetic identity -------------------------------------

def test_metrics_identity_is_exactly_zero():
    """Spec 10.1 item 1, tolerance 1e-9. Identical inputs must give
    NRMSE=0, VR=1, KL=0, KS=0, W1=0 -- not 'small', zero."""
    x = _mission_signal()
    r = compare_signal(x, x)
    assert abs(r["nrmse_range"]) < 1e-9, r["nrmse_range"]
    assert abs(r["nrmse_sigma"]) < 1e-9
    assert abs(r["bias"]) < 1e-9
    assert abs(r["vr"] - 1.0) < 1e-9, r["vr"]
    assert abs(r["vr_log"]) < 1e-9
    assert abs(r["kl_bits_real_vs_sim"]) < 1e-9, r["kl_bits_real_vs_sim"]
    assert abs(r["jsd_bits"]) < 1e-9
    assert abs(r["psi"]) < 1e-9
    assert abs(r["wasserstein"]) < 1e-9
    assert abs(r["ks_statistic"]) < 1e-9
    assert abs(r["pearson_r"] - 1.0) < 1e-9
    print("PASS: identity -> NRMSE/KL/JSD/PSI/W1/KS all 0, VR 1, rho 1 (<1e-9)")


def test_spectral_identity_lsd_is_zero():
    """Spec 10.1 item 1 also requires LSD = 0 on identical inputs."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(20000)
    s = compare_spectra(x, x, 1000.0)
    assert abs(s["lsd_db"]) < 1e-9, s["lsd_db"]
    assert abs(s["centroid_shift_hz"]) < 1e-9
    for band, ratio in s["band_ratios_db"].items():
        if ratio is not None:
            assert abs(ratio) < 1e-9, f"{band} ratio {ratio}"
    print("PASS: identity -> LSD 0 dB, centroid shift 0 Hz, every band ratio 0 dB")


def test_lag_identity_is_zero():
    """Spec 10.1 item 1: all lags 0 when the two logs are identical."""
    # The response must actually correlate with the command. An earlier
    # version used np.gradient(x), which is 90 degrees out of phase --
    # rho 0.248, and the 3.8 validity gate correctly refused it. That was
    # the test being wrong, not the gate.
    x = _mission_signal(n=4000)
    delayed = np.concatenate([np.zeros(4), x[:-4]])  # same 80 ms lag both sides
    lag = actuator_lag(x, delayed, x, delayed, 50.0, pair_name="identity")
    assert lag["delta_tau_ms"] == 0.0, lag
    assert lag["tau_real_ms"] == lag["tau_sim_ms"] != 0.0, lag
    print(f"PASS: same lag both sides -> delta_tau 0.0 ms "
          f"(tau {lag['tau_real_ms']} ms each, rho {lag['rho_real']:.3f})")


# --- 10.1 item 2: known offset -------------------------------------------

def test_known_offset_recovers_bias_exactly():
    """Spec 10.1 item 2: sim = real + 0.5 -> bias exactly 0.5, VR 1, rho 1,
    and KS equal to scipy's on the same arrays."""
    x = _mission_signal()
    y = x + 0.5
    r = compare_signal(x, y)
    assert abs(r["bias"] - 0.5) < 1e-9, r["bias"]
    assert abs(r["vr"] - 1.0) < 1e-9
    assert abs(r["pearson_r"] - 1.0) < 1e-9
    ref = _sp_stats.ks_2samp(x, y)
    assert r["ks_statistic"] == float(ref.statistic)
    print(f"PASS: +0.5 offset -> bias {r['bias']:.10f}, VR {r['vr']:.10f}, "
          f"rho {r['pearson_r']:.10f}, KS {r['ks_statistic']:.6f}")


# --- 10.1 item 3: known variance ratio -----------------------------------

def test_known_variance_ratio():
    """Spec 10.1 item 3: sim = real scaled x2 about its mean -> VR = 4.0,
    VR_log = 2.0, to 1e-9."""
    x = _mission_signal()
    y = (x - x.mean()) * 2.0 + x.mean()
    r = compare_signal(x, y)
    assert abs(r["vr"] - 4.0) < 1e-9, r["vr"]
    assert abs(r["vr_log"] - 2.0) < 1e-9, r["vr_log"]
    print(f"PASS: x2 amplitude -> VR {r['vr']:.10f}, VR_log {r['vr_log']:.10f}")


# --- 10.1 item 4: known lag ----------------------------------------------

def test_known_lag_60ms_on_200hz_chirp():
    """Spec 10.1 item 4: 60 ms delay on a 200 Hz chirp, recovered within
    +/-2 ms after parabolic refinement. 60 ms is 12 samples at 200 Hz, so
    an integer-only peak would pass -- the sub-sample case is covered by
    the 63 ms variant below, which is 12.6 samples."""
    fs, delay_s = 200.0, 0.060
    n = 8000
    t = np.arange(n) / fs
    cmd = sp_chirp = np.sin(2 * np.pi * (1.0 + 3.0 * t / t[-1]) * t)
    shift = int(round(delay_s * fs))
    resp = np.concatenate([np.zeros(shift), cmd[:-shift]])
    lag, rho = estimate_lag_s(cmd, resp, fs)
    err_ms = abs(lag * 1000.0 - 60.0)
    assert err_ms <= 2.0, f"recovered {lag*1000:.2f} ms, error {err_ms:.2f} ms"
    assert rho > 0.9
    print(f"PASS: 60 ms injected lag recovered as {lag*1000:.2f} ms "
          f"(err {err_ms:.2f} ms, rho {rho:.3f})")


def test_sub_sample_lag_needs_parabolic_refinement():
    """63 ms at 200 Hz is 12.6 samples -- not representable on the integer
    grid. This is the case parabolic refinement exists for, and it fails
    without it (nearest sample is 12 or 13 -> 60 or 65 ms, both >2 ms out)."""
    fs = 200.0
    n = 8000
    t = np.arange(n) / fs
    cmd = np.sin(2 * np.pi * 1.7 * t) + 0.5 * np.sin(2 * np.pi * 0.6 * t)
    # fractional delay via interpolation
    resp = np.interp(t - 0.063, t, cmd, left=0.0, right=0.0)
    lag, rho = estimate_lag_s(cmd, resp, fs)
    err_ms = abs(lag * 1000.0 - 63.0)
    assert err_ms <= 2.0, f"recovered {lag*1000:.2f} ms, error {err_ms:.2f} ms"
    print(f"PASS: 63 ms (12.6 samples) recovered as {lag*1000:.2f} ms, "
          f"err {err_ms:.2f} ms -- integer grid could not do this")


def test_lag_validity_gate_refuses_uncorrelated_pair():
    """Spec 3.8 [DECISION]: peak rho < 0.3 -> lag_unreliable, no number."""
    rng = np.random.default_rng(5)
    a, b = rng.standard_normal(4000), rng.standard_normal(4000)
    out = actuator_lag(a, b, a, b, 50.0, pair_name="noise")
    assert out["delta_tau_ms"] is None
    assert "lag_unreliable" in out["flags"]
    assert out["nulls"]["delta_tau_ms"] == "lag_unreliable"
    print(f"PASS: uncorrelated pair -> lag_unreliable, no number "
          f"(rho {out['rho_real']:.3f} < 0.3)")


# --- 10.1 item 5: known spectrum -----------------------------------------

def test_known_spectrum_rotor_band_recovers_injected_tone():
    """Spec 10.1 item 5: white noise vs white noise + a 40 Hz tone ->
    rotor-band (20-100 Hz) ratio recovers the injected tone power to
    within 1 dB."""
    fs, n = 1000.0, 200000
    rng = np.random.default_rng(7)
    t = np.arange(n) / fs
    noise = rng.standard_normal(n)
    amp = 0.5
    tone = amp * np.sin(2 * np.pi * 40.0 * t)
    real = noise
    sim = noise + tone

    f, p_r = welch_psd(real, fs)
    _, p_s = welch_psd(sim, fs)
    from app.services.reality_gap.spectral import band_power
    pr = band_power(f, p_r, 20.0, 100.0)
    ps = band_power(f, p_s, 20.0, 100.0)
    # Analytic: the tone adds amp^2/2 of power on top of the band's noise.
    expected_db = 10 * np.log10((pr + amp ** 2 / 2) / pr)
    ratios, nulls, _ = band_ratios_db(f, p_r, p_s, fs)
    err = abs(ratios["rotor"] - expected_db)
    assert err <= 1.0, f"measured {ratios['rotor']:.2f} dB vs expected {expected_db:.2f} dB"
    print(f"PASS: 40 Hz tone -> rotor band {ratios['rotor']:.2f} dB vs analytic "
          f"{expected_db:.2f} dB (err {err:.2f} dB)")


# --- 10.1 item 6: KL against a closed form -------------------------------

def test_kl_matches_gaussian_closed_form():
    """Spec 10.1 item 6 -- 'the test that proves the binning and smoothing
    are not silently distorting the number'. Two Gaussians, N=1e6, binned
    per 3.0, within 2% of the analytic KL converted to bits."""
    rng = np.random.default_rng(13)
    mu_p, sd_p = 0.0, 1.0
    mu_q, sd_q = 0.4, 1.3
    n = 1_000_000
    p_samples = rng.normal(mu_p, sd_p, n)
    q_samples = rng.normal(mu_q, sd_q, n)

    analytic_nats = (np.log(sd_q / sd_p)
                     + (sd_p ** 2 + (mu_p - mu_q) ** 2) / (2 * sd_q ** 2) - 0.5)
    analytic_bits = analytic_nats / np.log(2.0)

    edges, reason = freedman_diaconis_edges(np.concatenate([p_samples, q_samples]))
    assert edges is not None, reason
    P, Q = _smoothed_histograms(p_samples, q_samples, edges)
    measured = kl_divergence_bits(P, Q)
    rel = abs(measured - analytic_bits) / analytic_bits
    assert rel <= 0.02, f"measured {measured:.6f} bits vs analytic {analytic_bits:.6f} ({rel*100:.2f}%)"
    print(f"PASS: KL {measured:.6f} bits vs closed form {analytic_bits:.6f} bits "
          f"({rel*100:.2f}% < 2%), {edges.size-1} FD bins")


# --- 10.1 item 7: KS and Wasserstein match scipy exactly -----------------

def test_ks_and_wasserstein_match_scipy_exactly():
    """Spec 10.1 item 7, unchanged by A7: we call scipy, so the value must
    be scipy's, bit for bit. Guards argument-order and axis mistakes."""
    rng = np.random.default_rng(17)
    a = rng.normal(0, 1, 20000)
    b = rng.normal(0.3, 1.2, 20000)
    r = compare_signal(a, b)
    ks_ref = _sp_stats.ks_2samp(a, b)
    w_ref = _sp_stats.wasserstein_distance(a, b)
    assert r["ks_statistic"] == float(ks_ref.statistic), (r["ks_statistic"], ks_ref.statistic)
    assert r["ks_pvalue"] == float(ks_ref.pvalue)
    assert r["wasserstein"] == float(w_ref), (r["wasserstein"], w_ref)
    print(f"PASS: KS {r['ks_statistic']:.12f} and W1 {r['wasserstein']:.12f} "
          f"are scipy's values exactly (==, not approx)")


def test_kl_direction_is_real_vs_sim_and_asymmetric():
    """Spec 3.3 [DECISION]: P=real, Q=sim. Asymmetry proves the direction
    is actually honoured rather than the arguments being interchangeable."""
    rng = np.random.default_rng(19)
    a = rng.normal(0, 1, 50000)
    b = rng.normal(0.8, 2.0, 50000)
    edges, _ = freedman_diaconis_edges(np.concatenate([a, b]))
    P, Q = _smoothed_histograms(a, b, edges)
    fwd, rev = kl_divergence_bits(P, Q), kl_divergence_bits(Q, P)
    assert abs(fwd - rev) > 1e-3, (fwd, rev)
    assert compare_signal(a, b)["kl_bits_real_vs_sim"] == fwd
    print(f"PASS: KL(real||sim) {fwd:.4f} != KL(sim||real) {rev:.4f} bits; "
          f"compare_signal reports the real||sim direction")


# --- null-with-reason behaviour (spec 9, 10.6 item 29) -------------------

def test_degenerate_signal_nulls_not_zeros():
    """Spec 9 case 10: a stuck sensor nulls VR/KL/PSI/KS/W1 with reasons.
    Never 0.0 -- a fabricated zero reads as perfect agreement."""
    x = _mission_signal()
    stuck = np.full(x.size, 3.0)
    r = compare_signal(x, stuck)
    for key in ("vr", "kl_bits_real_vs_sim", "psi", "ks_statistic", "wasserstein"):
        assert r[key] is None, f"{key} should be null, got {r[key]}"
        assert key in r["nulls"], f"{key} null without a reason"
    assert "degenerate" in r["flags"]
    # The real side still has variance, so the sigma fallback survives.
    assert r["nrmse_sigma"] is not None
    print(f"PASS: stuck sim sensor -> VR/KL/PSI/KS/W1 null with reasons "
          f"{sorted(set(r['nulls'].values()))}, NRMSE_sigma still computed")


def test_constant_real_promotes_sigma_and_then_nulls_it():
    """Spec 3.1 [DECISION] + 9 case 10: range(real) below floor nulls
    NRMSE_range; a degenerate REAL side also kills the sigma fallback."""
    const = np.full(3000, 2.0)
    r = compare_signal(const, const + 0.1)
    assert r["nrmse_range"] is None and r["nulls"]["nrmse_range"] == "range_below_floor"
    assert r["nrmse_sigma"] is None and r["nulls"]["nrmse_sigma"] == "real_variance_degenerate"
    assert r["nrmse_primary"] == "nrmse_sigma"
    print("PASS: constant real -> nrmse_range null (range_below_floor), "
          "nrmse_sigma null (real_variance_degenerate), neither coerced to 0")


def test_low_rate_signal_gets_no_spectral_metrics():
    """Spec 9 case 5: below 5 Hz native -> all spectral metrics null."""
    x = _mission_signal(n=200)
    s = compare_spectra(x, x, 2.0, low_rate=True)
    assert s["lsd_db"] is None and s["nulls"]["lsd_db"] == "low_rate"
    assert all(v is None for v in s["band_ratios_db"].values())
    print("PASS: low_rate signal -> every spectral metric null with reason low_rate")


# --- A5 cutoff gate ------------------------------------------------------

def test_band_nulled_above_antialiasing_cutoff():
    """A5 / spec 3.7 item 2: a band whose LOWER edge sits above either
    log's cutoff is null with band_above_antialiasing_cutoff -- the exact
    real-hardware case, IMU_ACCEL_CUTOFF=30 Hz against a 20-100 Hz band's
    upper reach and a 100 Hz+ noise floor."""
    fs, n = 1000.0, 100000
    rng = np.random.default_rng(23)
    x = rng.standard_normal(n)
    f, p = welch_psd(x, fs)
    ratios, nulls, flags = band_ratios_db(f, p, p, fs, cutoff_real=30.0, cutoff_sim=30.0)
    assert ratios["noise_floor"] is None
    assert nulls["noise_floor"] == "band_above_antialiasing_cutoff"
    # rotor (20-100) straddles 30 Hz -> computed, but flagged
    assert ratios["rotor"] is not None
    assert any(f_.startswith("band_partially_filtered:rotor") for f_ in flags), flags
    print("PASS: 30 Hz cutoff -> noise_floor null (band_above_antialiasing_cutoff), "
          "rotor computed + band_partially_filtered")


def test_gate_below_band_lower_edge_nulls_rotor():
    """Spec 10.1 item 9b(g): both logs at IMU_ACCEL_CUTOFF=15 (below the
    rotor band's 20 Hz lower edge) -> rotor ratio null with reason."""
    fs, n = 1000.0, 100000
    rng = np.random.default_rng(29)
    x = rng.standard_normal(n)
    f, p = welch_psd(x, fs)
    ratios, nulls, _ = band_ratios_db(f, p, p, fs, cutoff_real=15.0, cutoff_sim=15.0)
    assert ratios["rotor"] is None
    assert nulls["rotor"] == "band_above_antialiasing_cutoff"
    print("PASS: 15 Hz cutoff (below 20 Hz edge) -> rotor null with "
          "band_above_antialiasing_cutoff")


def test_cutoff_mismatch_warns_but_still_computes():
    """A5 item 3 / spec 10.1 item 9b(b,c): mismatched cutoffs mark every
    spectral metric cutoff_mismatch, but the metrics are STILL computed --
    suppression applies to the hint (spec 6), not to the measurement."""
    fs, n = 1000.0, 100000
    rng = np.random.default_rng(31)
    x = rng.standard_normal(n)
    s = compare_spectra(x, x, fs, chain=ACCEL_CHAIN, cutoff_real=30.0, cutoff_sim=100.0)
    assert "cutoff_mismatch" in s["flags"]
    assert s["lsd_db"] is not None, "metric must still be computed"
    assert s["band_ratios_db"]["structural"] is not None
    matched = compare_spectra(x, x, fs, cutoff_real=100.0, cutoff_sim=100.0)
    assert "cutoff_mismatch" not in matched["flags"]
    print("PASS: 30 vs 100 Hz -> cutoff_mismatch flag, metrics still computed; "
          "matched 100/100 -> no flag")


def test_cutoff_zero_means_filter_disabled():
    """Spec 3.7 item 1: a cutoff of 0 in PX4 means DISABLED, not 0 Hz.
    Reading it literally would null every band on the widest-bandwidth log."""
    fs, n = 1000.0, 100000
    rng = np.random.default_rng(37)
    x = rng.standard_normal(n)
    f, p = welch_psd(x, fs)
    ratios, nulls, _ = band_ratios_db(f, p, p, fs, cutoff_real=0.0, cutoff_sim=0.0)
    assert ratios["noise_floor"] is not None, nulls
    assert ratios["rotor"] is not None
    print("PASS: cutoff 0.0 treated as disabled -> all bands measurable")


def test_bands_above_nyquist_are_null_on_the_50hz_grid():
    """Spec 3.7 (b): at the 50 Hz canonical grid Nyquist is 25 Hz, so the
    rotor and noise-floor bands are unavailable unless the 5.4 high-rate
    path is used. Null, not zero."""
    x = _mission_signal(n=20000)
    s = compare_spectra(x, x, 50.0)
    assert s["band_ratios_db"]["rotor"] is None
    assert s["band_ratio_nulls"]["rotor"] == "band_above_nyquist"
    assert s["band_ratios_db"]["noise_floor"] is None
    assert s["band_ratios_db"]["structural"] is not None
    print("PASS: 50 Hz grid -> rotor and noise_floor null (band_above_nyquist), "
          "rigid_body and structural computed")


# --- misc ----------------------------------------------------------------

def test_effective_sample_size_below_nominal_for_correlated_series():
    """Spec 3.6 [DECISION]: N_eff must shrink for autocorrelated data --
    that is the whole point of reporting it next to the p-value."""
    rng = np.random.default_rng(41)
    white = rng.standard_normal(10000)
    smooth = np.convolve(white, np.ones(50) / 50, mode="same")
    n_white = effective_sample_size(white)
    n_smooth = effective_sample_size(smooth)
    assert n_smooth < n_white / 5, (n_smooth, n_white)
    assert abs(n_white - 10000) < 10000 * 0.15
    print(f"PASS: N_eff white {n_white:.0f}/10000, smoothed {n_smooth:.0f} "
          f"({n_smooth/10000*100:.1f}% of nominal)")


def test_welch_nperseg_shrinks_for_short_signals():
    assert welch_nperseg(100000) == 1024
    short = welch_nperseg(2000)
    assert short <= 256 and (short & (short - 1)) == 0, short
    print(f"PASS: nperseg 1024 for long signals, {short} (power of two) for a short one")


def test_resource_panel_is_not_scored_and_nulls_absent_topics():
    """Spec 3.9 [DECISION]: context panel, never folded into a score."""
    panel = resource_panel(real_cpu=[0.4, 0.5, 0.9], sim_cpu=[0.1, 0.1, 0.2])
    assert panel["scored"] is False
    assert abs(panel["cpu_real_mean"] - 0.6) < 1e-9
    assert panel["mem_real_mean"] is None
    assert panel["nulls"]["mem_real_mean"] == "topic_absent"
    print(f"PASS: resource panel scored=False, cpu_real p95 "
          f"{panel['cpu_real_p95']:.2f}, absent topics null with reason")


PASS2A_TESTS = [
    test_metrics_identity_is_exactly_zero,
    test_spectral_identity_lsd_is_zero,
    test_lag_identity_is_zero,
    test_known_offset_recovers_bias_exactly,
    test_known_variance_ratio,
    test_known_lag_60ms_on_200hz_chirp,
    test_sub_sample_lag_needs_parabolic_refinement,
    test_lag_validity_gate_refuses_uncorrelated_pair,
    test_known_spectrum_rotor_band_recovers_injected_tone,
    test_kl_matches_gaussian_closed_form,
    test_ks_and_wasserstein_match_scipy_exactly,
    test_kl_direction_is_real_vs_sim_and_asymmetric,
    test_degenerate_signal_nulls_not_zeros,
    test_constant_real_promotes_sigma_and_then_nulls_it,
    test_low_rate_signal_gets_no_spectral_metrics,
    test_band_nulled_above_antialiasing_cutoff,
    test_gate_below_band_lower_edge_nulls_rotor,
    test_cutoff_mismatch_warns_but_still_computes,
    test_cutoff_zero_means_filter_disabled,
    test_bands_above_nyquist_are_null_on_the_50hz_grid,
    test_effective_sample_size_below_nominal_for_correlated_series,
    test_welch_nperseg_shrinks_for_short_signals,
    test_resource_panel_is_not_scored_and_nulls_absent_topics,
]
ALL_TESTS.extend(PASS2A_TESTS)

if __name__ == "__main__":
    _require_fixtures()
    failures = []
    for test in ALL_TESTS:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 -- report every failure, don't stop at the first
            failures.append((test.__name__, exc))
            print(f"FAIL: {test.__name__} -- {exc}")
    print()
    print(f"{len(ALL_TESTS) - len(failures)}/{len(ALL_TESTS)} D1 pass-1 tests passed.")
    if failures:
        sys.exit(1)
