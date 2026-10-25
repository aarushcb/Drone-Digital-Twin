"""
D1 Health, build steps 7-9: change detection, scoring, findings.

Run:  python3 tests/test_health_scoring.py

Standalone script (no pytest in this repo), same convention as the other
health tests. The real-log test skips when the gitignored fixture is absent.
"""

import copy
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from app.services.health import epochs as E  # noqa: E402
from app.services.health import findings as F  # noqa: E402
from app.services.health import norms as N  # noqa: E402
from app.services.health import report as R  # noqa: E402
from app.services.health import scoring as S  # noqa: E402
from app.services.health.baseline import Observation  # noqa: E402
from app.services.health.features import FEATURE_CATALOGUE, Feature  # noqa: E402
from app.services.health.windowing import FlightWindow  # noqa: E402

REAL = os.path.join(ROOT, "tests", "fixtures", "reality_gap", "real", "px4_777ed799.ulg")

# A healthy hex, values close to the real fixture's.
BASE = {
    "vib.metric_p50": 3.5, "vib.metric_p95": 8.0, "vib.clip_rate": 0.0, "vib.band_lo": -14.0,
    "vib.band_struct": -12.0, "vib.imbalance": -0.04, "act.sat_frac": 0.0, "act.mean": 0.49, "act.spread": 0.09,
    "act.hover": 0.50, "pwr.r_cell": 3.6, "pwr.v_cell_min": 3.75, "pwr.sag_hover": 0.29,
    "trk.rate_rms.roll": 0.74, "trk.rate_rms.pitch": 0.77, "trk.rate_rms.yaw": 0.49, "trk.att_err_p95": 6.4,
    "gps.eph_p50": 0.5, "gps.nsats_p10": 20.0, "gps.sacc_p50": 0.1, "gps.noise": 90.0, "gps.jamming": 30.0,
    "est.test_ratio_p95": 0.5,
}
JITTER = [0.0, 0.01, -0.01, 0.005, -0.005, 0.0075, -0.0075, 0.0025, -0.0025, 0.004, -0.004, 0.006]
WATCHED = {"MC_ROLLRATE_P": 0.08, "IMU_GYRO_CUTOFF": 40.0, "SYS_AUTOSTART": 6001.0,
           E.FIRMWARE_KEY: "37d06c4c"}


def window(qualified=True):
    return FlightWindow(qualified=qualified, reasons=[] if qualified else ["airborne_too_short"],
                        airborne_start_us=0, airborne_end_us=300_000_000, airborne_s=300.0,
                        airborne_source="land_detector", steady_s=120.0, steady_qualified=qualified)


def features(overrides=None, k=0):
    """Every catalogue feature; numeric ones jittered by +-1% so a history
    has a real (small) spread, R3 ones null as on a default-profile log."""
    vals = dict(BASE, **(overrides or {}))
    out = []
    for key, sub, tier, unit in FEATURE_CATALOGUE:
        if key in vals and vals[key] is not None:
            v = vals[key] * (1 + JITTER[k % len(JITTER)]) if vals[key] not in (0.0,) else 0.0
            detail = {"clip_events": 0} if key == "vib.clip_rate" else {}
            out.append(Feature(key, sub, tier, unit, value=v, detail=detail))
        elif key == "act.pattern":
            out.append(Feature(key, sub, tier, unit, label="balanced"))
        else:
            out.append(Feature(key, sub, tier, unit, reason="requires_fifo_accel" if key == "vib.band_hi"
                               else "no_per_cell_sensing"))
    return out


def history(n, overrides_at=None, pack_at=None, watched_at=None):
    flights = []
    for i in range(n):
        ov = (overrides_at or {}).get(i)
        flights.append(R.FlightInput(i + 1, "px4", features(ov, i), window(), pack_id=(pack_at or {}).get(i, 1),
                                     watched=dict((watched_at or {}).get(i, WATCHED))))
    return flights


def fkey(report, key):
    return next(f for f in report["features"] if f["key"] == key)


# ---------------------------------------------------------------- step 7: change detection

def test_reset_table_matches_spec():
    r = E.RESETS
    assert {k.split(".")[0] for k in r["props"]} == {"vib", "act", "pwr"} and r["props"] & E._prefixed("pwr.") == {"pwr.sag_hover"}
    assert r["pack"] == E._prefixed("pwr.")
    assert r["firmware_tuning"] == E._prefixed("trk.") | {"vib.band_lo", "vib.band_struct", "vib.band_hi"}
    assert r["frame"] == E.ALL_FEATURES - E._prefixed("gps.") and "est.test_ratio_p95" in r["frame"]
    assert r["payload"] == {"act.hover", "act.sat_frac", "pwr.sag_hover"} | E._prefixed("trk.")
    assert r["gps"] == E._prefixed("gps.", "est.")
    assert r["motors_escs"] == E._prefixed("vib.", "act.", "trk.")
    print("PASS: change type -> features reset matches the spec §2.4 table for all 7 types")


def test_watched_params_and_diff():
    params = {"MC_ROLLRATE_P": 0.08, "MC_PITCH_P": 6.0, "IMU_GYRO_CUTOFF": 40.0, "CA_ROTOR_COUNT": 6.0,
              "CA_ROTOR0_PX": 0.0, "CA_ROTOR7_PX": 0.0, "BAT1_N_CELLS": 12.0, "MPC_XY_VEL_MAX": 12.0}
    w = E.watched_params(params, "abc")
    assert "CA_ROTOR7_PX" not in w and "CA_ROTOR0_PX" in w and "MPC_XY_VEL_MAX" not in w and w[E.FIRMWARE_KEY] == "abc"
    cur = dict(w, MC_ROLLRATE_P=0.09, IMU_GYRO_CUTOFF=40.0 + 1e-7)
    cur[E.FIRMWARE_KEY] = "def"
    del cur["BAT1_N_CELLS"]                         # absent in one log: not a change
    ch = {c["param"]: c["change_type"] for c in E.param_changes(w, cur)}
    assert ch == {"MC_ROLLRATE_P": "firmware_tuning", E.FIRMWARE_KEY: "firmware_tuning"}, ch
    frame = E.param_changes(w, dict(w, CA_ROTOR0_PX=0.1))
    assert frame[0]["change_type"] == "frame"
    print("PASS: watched params skip unused rotor slots; gain + firmware change -> firmware_tuning; "
          "float noise and missing params are not changes; rotor geometry -> frame")


def test_param_detected_firmware_change_resets_tracking_and_bands_only():
    w2 = dict(WATCHED, MC_ROLLRATE_P=0.10)
    flights = history(8, watched_at={i: w2 for i in range(5, 8)})
    reps = R.analyse(flights)
    assert reps[5]["epoch_events"][0]["change_type"] == "firmware_tuning" and "new_epoch" in reps[5]["warnings"]
    assert reps[5]["param_diff"][0]["param"] == "MC_ROLLRATE_P"
    for f in reps[5]["features"]:
        if f["baseline"] is None and f["status"] in ("unscored", "usage"):
            continue
        reset = f["key"].startswith("trk.") or f["key"].startswith("vib.band_")
        assert (f["n"] == 1) == reset, (f["key"], f["n"])
    assert reps[5]["overall_score"] is not None     # 4 of 5 subsystems still scored
    print("PASS: a detected gain change resets trk.* and vib.band_* only (n back to 1), everything else keeps "
          "its baseline and the overall stays scored")


def test_declared_prop_change_absorbs_a_step_without_findings():
    """Spec §8.4, first half: a step after a declared prop change raises no finding."""
    step = {i: {"act.hover": 0.60} for i in range(6, 10)}
    reps = R.analyse(history(10, overrides_at=step), [E.ConfigEvent(6, "props", "user", note="new 15x5 props")])
    for r in reps[6:]:
        assert not [f for f in r["findings"] if f["key"] == "act.hover"], r["findings"]
        assert not r["change_prompts"]
    hov = fkey(reps[9], "act.hover")
    assert hov["n"] == 4 and hov["baseline"]["group"] == "epoch:0"
    print(f"PASS: hover 0.50 -> 0.60 after a declared prop change: no finding, no prompt; new baseline "
          f"(group {hov['baseline']['group']}, n {hov['n']})")


def test_undeclared_step_prompts_on_third_flight():
    """Spec §8.4, second half: the same step undeclared prompts on its 3rd flight."""
    step = {i: {"act.hover": 0.60} for i in range(6, 10)}
    reps = R.analyse(history(10, overrides_at=step))
    assert not reps[6]["change_prompts"] and not reps[7]["change_prompts"]
    p = reps[8]["change_prompts"]
    assert len(p) == 1 and p[0]["feature"] == "act.hover" and p[0]["direction"] == "up" and p[0]["flight_ids"] == [7, 8, 9]
    assert p[0]["text"] in reps[8]["verdict"]["lines"]
    assert any(f["key"] == "act.hover" and f["quarantined"] for f in reps[8]["findings"])
    assert fkey(reps[8], "act.hover")["admitted"] is False
    print(f"PASS: undeclared step -> quarantined, flagged, and on the 3rd flight: \"{p[0]['text']}\"")


def test_inferred_needs_consistency_and_one_direction():
    def obs(vals):
        return [Observation(i + 1, v, admitted=None) for i, v in enumerate(vals)]
    from app.services.health.baseline import process_history
    clean = [0.50, 0.505, 0.495, 0.50, 0.502, 0.498]
    scatter = process_history(obs(clean + [0.60, 0.75, 0.62]), "act.hover").observations
    assert E.inferred_change(scatter, "act.hover") is None          # 0.60 vs 0.75: |dz| = 7.5 > 2
    mixed = process_history(obs(clean + [0.60, 0.40, 0.60]), "act.hover").observations
    assert E.inferred_change(mixed, "act.hover") is None
    good = process_history(obs(clean + [0.60, 0.61, 0.60]), "act.hover").observations
    assert E.inferred_change(good, "act.hover").max_pairwise_z <= 2
    print("PASS: inferred prompt needs 3 consistent shifted flights in one direction; scatter or mixed don't prompt")


def test_power_keys_off_pack():
    reps = R.analyse(history(6, pack_at={i: 2 for i in (1, 3, 5)}))
    r6 = fkey(reps[5], "pwr.r_cell")
    assert r6["baseline"]["group"] == "pack:2" and r6["n"] == 3 and r6["baseline"]["flight_ids"] == [2, 4]
    assert fkey(reps[5], "vib.metric_p50")["n"] == 6
    reps = R.analyse(history(4), [E.ConfigEvent(2, "pack", "user")])
    assert fkey(reps[3], "pwr.r_cell")["n"] == 2 and fkey(reps[3], "vib.metric_p50")["n"] == 4
    print("PASS: power baselines are per pack (alternating packs -> separate groups); a declared pack change "
          "resets pwr.* only")


# ---------------------------------------------------------------- step 8: scoring

def test_blend_weights_and_baseline_score():
    assert [S.blend_weight(n) for n in range(1, 7)] == [0, 0, 1 / 3, 2 / 3, 1, 1]
    assert [S.baseline_score(z) for z in (0, 1, 2, 3)] == [100, 100, 85, 70] and abs(S.baseline_score(13 / 3) - 50) < 1e-9
    assert S.combine([60, 100]) == 0.5 * 60 + 0.5 * 80
    print("PASS: w = 0, 0, 1/3, 2/3, 1; S_baseline z 1/2/3 -> 100/85/70, ~4.3 -> 50; 0.5 min + 0.5 mean")


def test_flight_one_rules():
    r = R.analyse(history(1))[0]
    subs = r["subsystem_scores"]
    assert r["overall_score"] is None and r["score_band"] is None
    assert subs["IMU & vibration"] is not None and subs["Position & velocity"] is not None
    assert subs["Actuators"] is None and subs["Attitude & rates"] is None and subs["Power"] is None
    assert fkey(r, "pwr.r_cell")["status"] == "learning" and fkey(r, "pwr.v_cell_min")["status"] == "usage"
    assert r["checks"] and r["verdict"]["headline"].startswith("First flight logged: 9 of 9 checks OK")
    assert r["verdict"]["lines"] == ["Nothing outside PX4's published ranges.", "Fly 4 more to learn your drone's normal."]
    print("PASS: flight 1 -> overall null, Vibration and Position scored, Motors/Attitude/Power learning, "
          "power never scored, checklist verdict")


def test_overall_appears_on_flight_three():
    reps = R.analyse(history(3))
    assert reps[1]["overall_score"] is None and reps[2]["overall_score"] is not None
    hover = fkey(reps[2], "act.hover")
    assert hover["mode"] == "blended" and hover["weight"] == 1.0 and hover["n"] == 3
    vib = fkey(reps[2], "vib.metric_p50")
    assert abs(vib["score"] - round(vib["weight"] * vib["score_baseline"] + (1 - vib["weight"]) * vib["score_absolute"], 1)) <= 0.05
    assert reps[2]["checks"] is None and reps[2]["subsystems_scored"] == 5
    print(f"PASS: flight 3 -> overall {reps[2]['overall_score']} (all 5 subsystems), vibration blended "
          f"{vib['weight']:.3f}*{vib['score_baseline']} + ... = {vib['score']}, no checklist")


def test_absolute_bad_capped_forever():
    bad = {i: {"vib.metric_p50": 12.0} for i in range(10)}
    reps = R.analyse(history(10, overrides_at=bad))
    v = fkey(reps[9], "vib.metric_p50")
    assert v["score"] <= 49 and v["admitted"] is False
    f = next(x for x in reps[9]["findings"] if x["key"] == "vib.metric_p50")
    assert f["kind"] == "consistently_bad" and "every flight so far" in f["line"]
    assert reps[9]["findings"][0]["key"] == "vib.metric_p50"   # absolute bads sort first
    print(f"PASS: vibration 12 m/s² on all 10 flights -> score {v['score']} (<= 49), never learned as normal; "
          f"\"{f['text']}\"")


def test_absolute_bad_caps_a_blend():
    """The cap must bind when the baseline says 'normal': five admitted
    flights at 9.6 (just under the 9.81 bad edge), then 9.9 -- z < 1, so
    S_baseline = 100 and w = 1, yet the value is absolutely bad."""
    hist = history(6, overrides_at={i: {"vib.metric_p50": 9.6} for i in range(5)})
    hist[5].features = features({"vib.metric_p50": 9.9}, 0)
    v = fkey(R.analyse(hist)[5], "vib.metric_p50")
    assert v["score_baseline"] == 100.0 and v["weight"] == 1.0 and v["score_absolute"] < 50
    assert v["capped"] and v["score"] == 49.0
    print(f"PASS: S_baseline {v['score_baseline']} (normal for this drone) but absolutely bad -> capped at "
          f"{v['score']} in personal mode")


def test_overall_needs_four_subsystems():
    gone = {k: None for k in BASE if k.startswith(("trk.", "gps.", "est."))}
    reps = R.analyse(history(5, overrides_at={i: gone for i in range(5)}))
    assert reps[4]["subsystems_scored"] == 3 and reps[4]["overall_score"] is None
    print("PASS: 3 of 5 subsystems scored -> overall null (spec §4.2)")


def test_definitional_raises_finding_without_scoring():
    r = R.analyse(history(1, overrides_at={0: {"act.sat_frac": 0.012, "est.test_ratio_p95": 1.2}}))[0]
    kinds = {f["key"]: f["kind"] for f in r["findings"]}
    assert kinds == {"act.sat_frac": "definitional", "est.test_ratio_p95": "definitional"}, kinds
    assert r["subsystem_scores"]["Actuators"] is None
    sat = next(f for f in r["findings"] if f["key"] == "act.sat_frac")
    assert sat["confidence"] == "high" and sat["text"] == "Motors at full output 1.2% of the flight"
    print("PASS: saturation and estimator rejection raise high-confidence findings on flight 1 without "
          "scoring their subsystem")


# ---------------------------------------------------------------- step 9: findings

def test_every_line_fits_nine_words():
    from app.services.health.scoring import FeatureScore
    from app.services.health.baseline import Baseline
    checked = 0
    for key, d in F.DISPLAY.items():
        sub, tier, unit = next((s, t, u) for k, s, t, u in FEATURE_CATALOGUE if k == key)
        for v in (0.001, 1.234, 12.345, 123.4, 9999.9):
            feat = Feature(key, sub, tier, unit, value=v, detail={"clip_events": 140})
            base = Baseline("rolling", median=v / 2, mad=0.1, spread=0.2, n=7)
            fs = FeatureScore(feat, "epoch:initial", 8, 8, "personal", 1.0,
                              N.absolute_score(key, v, "px4", feat.detail), rolling=base, z=5.0)
            assert F._words(F._personal_line(d, fs)) <= 9
            if fs.absolute.norm is not None and fs.absolute.norm.kind != "usage":
                assert F._words(F._absolute_line(d, fs)) <= 9
                assert F._words(F._consistent_line(d, fs)) <= 9
            checked += 1
    print(f"PASS: personal/absolute/consistently-bad lines <= 9 words for {checked} feature x value cases")


def test_no_finding_without_a_real_value():
    from app.services.health.scoring import FeatureScore
    feat = Feature("vib.metric_p50", "IMU & vibration", "R1", "m/s^2", reason="signal_absent:vibe")
    fs = FeatureScore(feat, "g", 1, 1, "absolute", 0.0, N.absolute_score("vib.metric_p50", None, "px4"))
    assert F.finding_for(fs) is None
    fs.feature = Feature("vib.metric_p50", "IMU & vibration", "R1", "m/s^2", value=float("nan"))
    try:
        F._make(fs, "absolute", "Vibration nan", None, "low")
        raise AssertionError("NaN figure produced a finding")
    except ValueError:
        pass
    print("PASS: null value -> no finding; a NaN figure can't construct one")


def test_confidence_and_hedges_follow_spec():
    reps = R.analyse(history(12, overrides_at={11: {"vib.metric_p50": 6.5, "trk.rate_rms.roll": 1.2}}))
    f = {x["key"]: x for x in reps[11]["findings"]}
    assert f["vib.metric_p50"]["confidence"] == "high" and f["vib.metric_p50"]["text"] == f["vib.metric_p50"]["line"]
    assert f["trk.rate_rms.roll"]["confidence"] == "medium" and f["trk.rate_rms.roll"]["text"].startswith("Looks like ")
    early = R.analyse(history(1, overrides_at={0: {"vib.metric_p50": 6.0}}))[0]["findings"][0]
    assert early["confidence"] == "low" and early["text"].startswith("Early read: ")
    assert all(c["phrase"].startswith("Worth checking") for c in early["causes"])
    assert all(c["phrase"].startswith(("Most often", "Also check")) for c in f["vib.metric_p50"]["causes"])
    print("PASS: personal R1 n>=10 -> high (no hedge); R2 personal -> medium 'Looks like'; borrowed norm -> "
          "low 'Early read' with 'Worth checking' causes")


def test_every_finding_carries_the_required_fields():
    reps = R.analyse(history(6, overrides_at={5: {"vib.metric_p50": 5.5, "act.hover": 0.6}}))
    fs = reps[5]["findings"]
    assert fs and len(reps[5]["verdict"]["lines"]) <= 3
    for f in fs:
        for k in ("figure", "comparison", "mode", "n", "confidence", "causes"):
            assert f[k] not in (None, [], ""), (f["key"], k)
        assert [c["rank"] for c in f["causes"]] == list(range(1, len(f["causes"]) + 1))
    scores = [f["score"] for f in fs]
    assert scores == sorted(scores), scores
    print(f"PASS: {len(fs)} findings, each with figure/comparison/mode/n/confidence/ranked causes, worst first")


def test_pattern_finding_names_the_motor():
    """Spec §5.1 pattern template: motor 3's excess over the mean of the rest."""
    per = [0.48, 0.48, 0.56, 0.48, 0.48, 0.48]
    flights = history(6, overrides_at={5: {"act.spread": 0.16}})
    for f in flights[5].features:
        if f.key == "act.pattern":
            f.label, f.detail = "single_motor_high", {"motor_number": 3, "high_motor_numbers": [3]}
        if f.key == "act.mean":
            f.detail = {"per_motor": per}
    f = next(x for x in R.analyse(flights)[5]["findings"] if x["key"] == "act.spread")
    assert f["kind"] == "pattern" and f["line"] == "Motor 3 working 17% harder than the others", f["line"]
    assert f["causes"][0]["text"].startswith("that motor or prop producing less thrust")
    assert len(f["causes"]) >= 2                      # §6.2: ranked candidates, never one named part
    print(f"PASS: spread step with one motor high -> \"{f['text']}\" with ranked prop/motor/ESC/arm causes")


# §6 claims a finding must never be able to make, one or more per item.
FORBIDDEN_SAMPLES = [
    "Motor 3 has about 40 hours left", "Props near end of life", "Remaining life: 120 cycles",   # 6.1
    "Motor 3 is faulty", "The ESC has failed", "Diagnosis: bad bearing", "The cause is prop 2",     # 6.2
    "Battery health 82%", "Pack at 64% capacity", "State of health is low", "remaining capacity 3.1 Ah",  # 6.3
    "Prop 2 unbalanced at 120 Hz", "Blade-pass vibration up",                                         # 6.4
    "Motor 3 RPM low", "Motor temperature high",                                                      # 6.5
    "Safe to fly", "Your drone is airworthy", "Unsafe to fly today",                                  # 6.6
    "Vibration better than average", "Worse than most drones", "Top percentile of the fleet",         # 6.7
    "Tracking error due to wind", "It's mechanical, not wind",                                        # 6.8
    "Vibration rose because you swapped props",                                                       # 6.9
    "Your PIDs are bad", "Badly tuned roll axis",                                                     # 6.10
    "Drone is 300 g heavier", "Payload weighs 2 kg",                                                  # 6.11
    "Same as ArduPilot VIBE",                                                                         # 6.12
    "Simulator fidelity 80%",                                                                         # 6.13
]


def test_section6_claims_cannot_be_emitted():
    blocked = 0
    for text in FORBIDDEN_SAMPLES:
        try:
            F.check_claim(text)
        except F.ForbiddenClaim:
            blocked += 1
            continue
        raise AssertionError(f"§6 claim passed the check: {text!r}")
    # The constructor itself refuses: a finding whose line makes a §6 claim can't exist.
    from app.services.health.scoring import FeatureScore
    feat = Feature("pwr.r_cell", "Power", "R2", "mOhm", value=4.1)
    fs = FeatureScore(feat, "pack:1", 6, 6, "personal", 1.0, N.absolute_score("pwr.r_cell", 4.1, "px4"))
    for line in ("Battery health 82%", "Pack has 30 cycles left", "Safe to fly"):
        try:
            F._make(fs, "personal", line, None, "high")
            raise AssertionError(f"_make accepted {line!r}")
        except F.ForbiddenClaim:
            pass
    # A report carrying a forbidden key or string is rejected as a whole.
    rep = R.analyse(history(1))[0]
    for bad in ({"remaining_life_hours": 5, "remaining_life": 5}, {"safe_to_fly": True}):
        try:
            F.assert_report_honest(dict(rep, **bad))
            raise AssertionError(f"report with {bad} passed")
        except F.ForbiddenClaim:
            pass
    tampered = copy.deepcopy(rep)
    tampered["verdict"]["lines"].append("Motor 4 is failing")
    try:
        F.assert_report_honest(tampered)
        raise AssertionError("tampered verdict passed")
    except F.ForbiddenClaim:
        pass
    print(f"PASS: all {blocked} §6 sample claims blocked (13 items); the Finding constructor refuses them; "
          "forbidden keys and edited verdict strings fail the whole report")


def test_no_false_positives_on_real_output():
    """Every string the generator can produce passes -- the guard isn't so
    broad that it blocks honest findings."""
    scenarios = [history(1), history(6, overrides_at={5: {k: v * 1.5 for k, v in BASE.items() if v}}),
                 history(6, overrides_at={5: {"gps.nsats_p10": 4.0, "gps.eph_p50": 3.4, "gps.sacc_p50": 0.6,
                                              "vib.imbalance": 31.0, "act.sat_frac": 0.02}})]
    n = 0
    for h in scenarios:
        for r in R.analyse(h):
            F.assert_report_honest(r)
            n += len(r["findings"])
    print(f"PASS: {n} findings across 13 reports, including GPS gates, trigger-only and definitional, pass the §6 guard")


def test_real_fixture_flight_one():
    if not os.path.exists(REAL):
        print("SKIP: real fixture absent")
        return
    from app.services.reality_gap.parsers import parse_log
    from app.services.health.features import extract_features
    from app.services.health.windowing import window_flight
    p = parse_log(REAL)
    w = window_flight(p)
    rep = R.analyse([R.FlightInput(1, "px4", extract_features(p, w), w, pack_id=1,
                                   watched=E.watched_params(p.params, p.meta.firmware_version))])[0]
    assert rep["overall_score"] is None and rep["subsystem_scores"]["IMU & vibration"] is not None
    assert [f["key"] for f in rep["findings"]] == ["vib.metric_p50"]
    assert rep["findings"][0]["text"] == "Early read: Vibration 5.5 m/s², above PX4's 4.9 guide"
    assert rep["verdict"]["headline"] == "First flight logged: 8 OK, 1 to look at"
    assert {g["feature"] for g in rep["logging_gaps"]} == {"vib.band_hi", "pwr.cell_delta"}
    print(f"PASS: real hex as flight 1 -> \"{rep['verdict']['headline']}\" / \"{rep['findings'][0]['text']}\"")


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"FAIL: {t.__name__} -- {exc!r}")
    print(f"\n{passed}/{len(tests)} D1 health step 7-9 tests passed.")
    sys.exit(0 if passed == len(tests) else 1)


if __name__ == "__main__":
    main()
