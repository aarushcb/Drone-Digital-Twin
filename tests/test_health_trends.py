"""
D1 Health, build step 13: Trends on raw feature values (spec §4.4).

Run:  python3 tests/test_health_trends.py
"""

import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from app.services.health import epochs as E  # noqa: E402
from app.services.health import report as R  # noqa: E402
from app.services.health import trends as T  # noqa: E402
from tests.test_health_scoring import WATCHED, features, window  # noqa: E402


def flights(values, key="act.hover", extra=None, packs=None, when=None):
    """One flight per value of `key`, everything else constant (no jitter)."""
    out = []
    for i, v in enumerate(values):
        ov = dict({key: v}, **((extra or {}).get(i, {})))
        out.append(R.FlightInput(i + 1, "px4", features(ov, 0), window(), pack_id=(packs or {}).get(i, 1),
                                 watched=dict(WATCHED), when=(when or {}).get(i)))
    return out


def series(tr, key, pack=None):
    return next(s for c in tr["cards"] for s in c["series"] if s["feature"] == key and (pack is None or s["pack_id"] == pack))


def test_drift_fires_while_every_report_stays_calm():
    """The case Trends exists for: hover creeping up 0.006 per flight. Each
    flight sits close to a rolling median that climbs with it, so no report
    flags it; the median of the last 5 has left the frozen reference band."""
    vals = [0.50] * 5 + [0.50 + 0.006 * k for k in range(1, 11)]
    hist = flights(vals)
    for rep in R.analyse(hist):
        assert not [f for f in rep["findings"] if f["key"] == "act.hover"], (rep["run"], rep["findings"])
    s = series(T.build(hist), "act.hover")
    assert s["reference"]["median"] == 0.5 and abs(s["reference"]["high"] - 0.54) < 1e-9
    drift = [c for c in s["callouts"] if c["kind"] == "drift"]
    assert drift and "crept up from 50% to" in drift[0]["text"], s["callouts"]
    assert s["latest_delta"]["figure"] == "+6%"
    # The latest flight's own score is calm (fine), but the pill sits next to
    # "has crept up", so it may not read greener than "watch".
    assert s["latest_delta"]["score"] >= 85 and s["latest_delta"]["band"] == "watch"
    print(f"PASS: every per-flight report calm, yet Trends says \"{drift[0]['text']}\"")


def test_pill_band_is_never_greener_than_its_callouts():
    assert T.pill_band(95.0, []) == "fine"
    assert T.pill_band(95.0, [{"kind": "drift"}]) == "watch"
    assert T.pill_band(60.0, [{"kind": "step"}]) == "off"        # already worse: unchanged
    assert T.pill_band(None, [{"kind": "guide"}]) == "watch"
    assert T.pill_band(None, []) is None
    print("PASS: pill band = the score's band, floored at 'watch' by any Step/Drift/Guide callout")


def test_step_fires_on_three_flights_outside_the_band():
    s = series(T.build(flights([0.50] * 6 + [0.60] * 3)), "act.hover")
    step = [c for c in s["callouts"] if c["kind"] == "step"]
    assert step and step[0]["text"].startswith("Hover throttle stepped up: 60%, 60% and 60%"), s["callouts"]
    two = series(T.build(flights([0.50] * 6 + [0.60] * 2)), "act.hover")
    assert not [c for c in two["callouts"] if c["kind"] == "step"]
    print(f"PASS: 3 flights outside the band -> \"{step[0]['text']}\"; 2 flights -> no Step")


def test_declared_change_is_a_labelled_step_not_a_fault():
    vals = [3.5] * 5 + [4.8] * 4      # band high is 3.5 + 2 x 0.5 = 4.5
    hist = flights(vals, key="vib.metric_p50", when={5: "2026-10-12T09:00:00"})
    s = series(T.build(hist, [E.ConfigEvent(5, "props", "user")]), "vib.metric_p50")
    assert s["boundaries"] == [{"flight_number": 6, "change_type": "props", "source": "user", "label": "New props, 12 Oct"}]
    assert [p["current"] for p in s["points"]] == [False] * 5 + [True] * 4          # history before it greyed
    assert s["reference"] is None and s["reference_progress"] == {"have": 4, "need": 5}   # the band restarts
    assert not s["callouts"]
    undeclared = series(T.build(hist), "vib.metric_p50")
    assert {c["kind"] for c in undeclared["callouts"]} >= {"step"}               # the same jump, undeclared, is flagged
    print("PASS: declared prop change -> boundary 'New props, 12 Oct', earlier points greyed, band restarts "
          "(4 of 5), no callout; undeclared it raises Step")


def test_guide_works_before_a_reference_exists():
    s = series(T.build(flights([5.3, 5.4, 5.2], key="vib.metric_p50")), "vib.metric_p50")
    assert s["reference"] is None and s["reference_progress"]["have"] == 3
    assert [g["value"] for g in s["guides"]] == [4.905, 9.81] and s["guides"][0]["label"] == "PX4 guide"
    g = [c for c in s["callouts"] if c["kind"] == "guide"]
    assert len(s["callouts"]) == 1 and g and g[0]["label"] == "vs PX4 guide"
    assert g[0]["text"] == "Vibration past PX4's 4.9 guide on the last 3 flights"
    print(f"PASS: 3 flights past PX4's line with no reference yet -> Guide only: \"{g[0]['text']}\"")


def test_what_is_and_is_not_trended():
    hist = flights([0.5] * 3, packs={1: 2})
    tr = T.build(hist)
    keys = {s["feature"] for c in tr["cards"] for s in c["series"]}
    assert keys == {"vib.metric_p50", "vib.imbalance", "act.spread", "act.hover", "pwr.r_cell"}, keys
    assert {s["label"] for c in tr["cards"] for s in c["series"] if s["feature"] == "pwr.r_cell"} == \
        {"Pack resistance · pack 1", "Pack resistance · pack 2"}                    # one line per pack
    assert [n["title"] for n in tr["not_trended"]] == ["Attitude & control", "Position"]
    assert series(tr, "act.hover")["caveat"] and series(tr, "act.hover")["min_span"] == 4 * 0.02
    print("PASS: vibration, imbalance, spread, hover and resistance (one line per pack) trended; "
          "tracking, GPS, estimator, min cell and sag not")


def test_clipping_shows_as_event_markers():
    hist = flights([3.5] * 3, key="vib.metric_p50")
    for f in hist[1].features:
        if f.key == "vib.clip_rate":
            f.value, f.detail = 1.2, {"clip_events": 12}
    card = next(c for c in T.build(hist)["cards"] if c["key"] == "vibration")
    assert card["markers"] == [{"flight_number": 2, "kind": "clipping", "events": 12,
                                "text": "12 accelerometer clipping events"}]
    print("PASS: clipping is a marker on the vibration card (flight 2, 12 events), not a line")


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
    print(f"\n{passed}/{len(tests)} D1 health step 13 trends tests passed.")
    sys.exit(0 if passed == len(tests) else 1)


if __name__ == "__main__":
    main()
