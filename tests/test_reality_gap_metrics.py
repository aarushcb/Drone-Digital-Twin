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
