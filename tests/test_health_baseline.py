"""
D1 Health, build steps 4-6: published thresholds and absolute scoring, the
baseline engine, and the schema migration.

Run:  python3 tests/test_health_baseline.py

Standalone script (no pytest in this repo), same convention as the other
test files. The migration test runs Alembic against a temporary SQLite file
with DATABASE_URL set explicitly -- it never touches Neon.
"""

import os
import sqlite3
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from app.services.health import baseline as B  # noqa: E402
from app.services.health import norms as N  # noqa: E402

CLEAN = [5.2, 5.6, 5.4, 5.3, 5.5, 5.45, 5.35, 5.6]   # healthy vibration p50 flights, m/s^2
BAD = 14.0                                             # absolutely bad vibration (above 9.81)


def obs(values, start_id=1, absolute_bad_at=(), unqualified_at=(), invalid_at=()):
    out = []
    for i, v in enumerate(values):
        fid = start_id + i
        out.append(B.Observation(fid, v, qualified=fid not in unqualified_at, valid=fid not in invalid_at,
                                 absolute_bad=fid in absolute_bad_at))
    return out


# ---------------------------------------------------------------- step 4: norms

def test_every_norm_is_sourced_dated_and_rated():
    for n in N.NORMS:
        assert n.source and n.url.startswith("https://"), n
        assert n.checked == "2026-10-25", n
        assert n.confidence in ("low", "medium", "high"), n
    print(f"PASS: all {len(N.NORMS)} norms carry a source, URL, check date and confidence")


def test_graded_edges_are_exact():
    """The graded curve itself (spec §4.1), still used by any GRADED norm."""
    g = N.graded_score
    assert g(0, 4.905, 9.81) == 100 and abs(g(4.905, 4.905, 9.81) - 85) < 1e-9
    assert abs(g(9.81, 4.905, 9.81) - 50) < 1e-9 and g(19.62, 4.905, 9.81) == 0
    assert abs(g(7.3575, 4.905, 9.81) - 67.5) < 1e-9     # halfway between edges -> halfway between 85 and 50
    print("PASS: graded curve 0/healthy/bad/2xbad -> 100/85/50/0")


def test_vibration_level_is_context_never_a_cold_start_score():
    """Amended 2026-10-27: the 4.9/9.81 lines judged 65%/30% of 23 surveyed real
    drones; the level is reported as context, never scored or 'absolutely bad'."""
    for autopilot, v in (("px4", 5.47), ("px4", 20.0), ("ardupilot", 75.0)):
        r = N.absolute_score("vib.metric_p50", v, autopilot)
        assert r.score is None and r.status == "context", (autopilot, v, r)
        assert not N.is_absolute_bad("vib.metric_p50", v, autopilot)    # so it can always join a baseline
    c = N.norm_for("vib.metric_p50", "px4").context
    assert (c["vehicles"], c["above_healthy"], c["above_bad"]) == (23, 15, 7)
    print("PASS: vibration level is context on both autopilots; never scored, never blocks the baseline")

def test_clipping_and_the_imbalance_trigger_are_facts_not_scores():
    """Clipping (borrowed ArduPilot '<100 likely ok') and PX4's imbalance
    trigger are reported from flight one as definitional facts (spec §3.2)."""
    from app.services.health.features import Feature
    from app.services.health import scoring as S
    for key in ("vib.clip_rate", "vib.imbalance"):
        assert N.norm_for(key, "px4") is None and N.norm_for(key, "ardupilot") is None, key
    clip = lambda n: S.definitional(Feature("vib.clip_rate", "vib", "R1", "1/min", value=n / 10.0,
                                            detail={"clip_events": n}))
    assert clip(0) is None and clip(1) == "accel_clipping"
    imb = lambda v: S.definitional(Feature("vib.imbalance", "vib", "R2", "", value=v))
    assert imb(29.9) is None and imb(30.0) == "imbalance_detector"
    print("PASS: any clipping, or PX4's imbalance trigger, is a fact from flight one; neither is a cold-start score")

def test_gates_pass_or_fail_without_invented_severity():
    assert N.absolute_score("gps.eph_p50", 2.9, "px4").score == 100
    assert N.absolute_score("gps.eph_p50", 3.1, "px4").score == N.GATE_FAIL_SCORE
    assert N.absolute_score("gps.nsats_p10", 6, "px4").score == 100
    assert N.absolute_score("gps.nsats_p10", 5, "px4").score == N.GATE_FAIL_SCORE
    assert N.absolute_score("gps.eph_p50", 50.0, "px4").score == N.GATE_FAIL_SCORE   # no grading past the edge
    print("PASS: single-edge GPS gates -> 100 pass / 70 fail, no invented second edge")


def test_usage_norms():
    for v, status in ((3.5, "usage_flown_to_empty"), (3.75, "usage_ok")):
        r = N.absolute_score("pwr.v_cell_min", v, "px4")
        assert r.score is None and r.status == status, r
    assert not N.is_absolute_bad("pwr.v_cell_min", 2.0, "px4")   # usage never counts as a health verdict
    print("PASS: min cell voltage never scored (usage note only)")

def test_no_norm_means_no_absolute_score():
    for key in ("act.spread", "trk.att_err_p95", "pwr.r_cell", "vib.band_struct"):
        assert N.absolute_score(key, 1.0, "px4").status == "no_norm", key
    assert N.autopilot_of("ulog") == "px4" and N.autopilot_of("dataflash") == "ardupilot"
    print("PASS: features without a published norm get no absolute score (baseline-only, spec 3.2)")


# ---------------------------------------------------------------- step 5: baseline engine

def test_early_outliers_cannot_steer_the_median():
    """Bad flights before the z gate exists (n < 5): only the median protects."""
    for bad_positions in ((0,), (0, 2)):
        values = list(CLEAN)
        for p in bad_positions:
            values.insert(p, 9.5)   # far from the clean flights but below the 9.81 absolute-bad edge, so gate (c) can't catch it
        res = B.process_history(obs(values), "vib.metric_p50")
        for base in (res.rolling, res.reference):
            assert min(CLEAN) <= base.median <= max(CLEAN), (bad_positions, base.kind, base.median)
    print(f"PASS: 1 and 2 bad flights at the start -> rolling and reference medians stay inside "
          f"the clean range [{min(CLEAN)}, {max(CLEAN)}]")


def test_late_outliers_are_quarantined_not_learned():
    values = CLEAN[:6] + [11.0, 11.5] + CLEAN[6:]
    res = B.process_history(obs(values), "vib.metric_p50")
    rejected = [o for o in res.observations if not o.admitted]
    assert [o.flight_id for o in rejected] == [7, 8], [(o.flight_id, o.quarantine_reason) for o in rejected]
    assert all(o.quarantine_reason.startswith("outlier:modified_z=") for o in rejected)
    assert all(o.flight_id not in res.rolling.flight_ids for o in rejected)
    assert min(CLEAN) <= res.rolling.median <= max(CLEAN)
    print(f"PASS: 2 bad flights after 6 clean -> quarantined ({rejected[0].quarantine_reason}), "
          f"kept but not learned; median {res.rolling.median:.2f} inside the clean range")


def test_absolutely_bad_first_flight_never_joins():
    res = B.process_history(obs([BAD] + CLEAN, absolute_bad_at={1}), "vib.metric_p50")
    first = res.observations[0]
    assert not first.admitted and first.quarantine_reason == "absolute_bad"
    assert 1 not in res.reference.flight_ids
    print("PASS: an absolutely bad first flight is quarantined (gate c), even before any baseline exists")


def test_gate_rejects_unqualified_and_invalid():
    res = B.process_history(obs(CLEAN[:4], unqualified_at={2}, invalid_at={3}), "vib.metric_p50")
    reasons = {o.flight_id: o.quarantine_reason for o in res.observations}
    assert reasons[2] == "flight_not_qualified" and reasons[3] == "feature_invalid"
    assert res.mode == "absolute"   # only 2 admitted
    print("PASS: unqualified flight and invalid feature are rejected with their reasons")


def test_unexplained_improvement_is_also_quarantined():
    res = B.process_history(obs(CLEAN[:6] + [1.0]), "vib.metric_p50")
    assert not res.observations[-1].admitted
    print("PASS: a large unexplained improvement is quarantined too (two-sided gate, spec 2.4 asks about it)")


def test_one_sided_z_and_floor():
    base = B.Baseline("rolling", median=5.0, mad=0.0, spread=max(0.0, B.SIGMA_FLOOR["vib.metric_p50"]), n=5)
    assert base.spread == 0.5                                   # identical history -> the floor, not zero
    assert B.robust_z(6.0, base, B.HIGHER_IS_WORSE) == 2.0
    assert B.robust_z(4.0, base, B.HIGHER_IS_WORSE) == 0.0      # harmless direction scores 0
    sats = B.Baseline("rolling", median=20, mad=1, spread=1.4826, n=5)
    assert B.robust_z(17, sats, B.LOWER_IS_WORSE) > 2 and B.robust_z(23, sats, B.LOWER_IS_WORSE) == 0
    flat = B.process_history(obs([5.0] * 6), "vib.metric_p50")
    assert flat.rolling.mad == 0 and flat.rolling.spread == 0.5
    print("PASS: one-sided z (harmless direction = 0); MAD of 0 falls back to the 0.5 m/s^2 floor")


def test_reference_freezes_while_rolling_follows_drift():
    drift = [5.0, 5.1, 5.0, 5.05, 4.95] + [5.0 + 0.3 * k for k in range(1, 9)]   # slow climb after flight 5
    res = B.process_history(obs(drift), "vib.metric_p50")
    assert res.reference.flight_ids == [1, 2, 3, 4, 5] and abs(res.reference.median - 5.0) < 1e-9
    assert res.rolling.median > 5.9
    z_ref = B.robust_z(drift[-1], res.reference, B.HIGHER_IS_WORSE)
    z_roll = B.robust_z(drift[-1], res.rolling, B.HIGHER_IS_WORSE)
    assert z_ref > z_roll
    print(f"PASS: slow drift -> frozen reference keeps median 5.00 (z {z_ref:.1f}) while rolling "
          f"follows it (z {z_roll:.1f}); why Trends uses the reference")


def test_modes_and_groups():
    assert [B.mode_for(n) for n in range(7)] == ["absolute"] * 3 + ["blended"] * 2 + ["personal"] * 2
    assert B.baseline_group("pwr.r_cell", None, 7) == "pack:7"
    assert B.baseline_group("pwr.r_cell", 3, 7) == "pack:7/epoch:3"
    assert B.baseline_group("vib.metric_p50", 3, 7) == "epoch:3"
    assert B.baseline_group("vib.metric_p50", None, 7) == "epoch:initial"
    assert B.baseline_group("pwr.sag_hover", None, None) == "pack:unknown"
    print("PASS: 0-2 absolute, 3-4 blended, 5+ personal; power features grouped per pack")


def test_every_baselined_feature_has_a_floor_and_direction():
    from app.services.health.features import FEATURE_CATALOGUE
    scored = [k for k, _, tier, _ in FEATURE_CATALOGUE if tier != "R3" and k not in B.NOT_BASELINED]
    missing = [k for k in scored if k not in B.SIGMA_FLOOR or k not in B.DIRECTION]
    assert not missing, missing
    print(f"PASS: all {len(scored)} baselined features have a spread floor and a harmful direction")


# ---------------------------------------------------------------- step 6: migration

def test_migration_upgrades_and_downgrades_on_sqlite():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "health.db")
        env = dict(os.environ, DATABASE_URL=f"sqlite:///{db}")
        alembic = [sys.executable, "-m", "alembic"]
        run = lambda *args: subprocess.run(alembic + list(args), cwd=ROOT, env=env, capture_output=True, text=True)
        tables = lambda: {r[0] for r in sqlite3.connect(db).execute("select name from sqlite_master where type='table'")}
        health = {"flights", "flight_features", "feature_baselines", "config_epochs", "packs"}

        up = run("upgrade", "head")
        assert up.returncode == 0, up.stderr[-500:]
        assert health <= tables()
        down = run("downgrade", "-1")
        assert down.returncode == 0, down.stderr[-500:]
        assert not (health & tables()) and {"drones", "telemetry", "users"} <= tables()
        assert run("upgrade", "head").returncode == 0
    print("PASS: migration upgrades, downgrades (leaving existing tables intact) and re-upgrades on SQLite")


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL: {t.__name__} -- {exc!r}")
    print(f"\n{passed}/{len(tests)} D1 health step 4-6 tests passed.")
    sys.exit(0 if passed == len(tests) else 1)


if __name__ == "__main__":
    main()
