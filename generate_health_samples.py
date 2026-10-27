"""
Generate D1 health sample reports from the real fixture log.

  python3 generate_health_samples.py [out_dir]

Writes health-report-flight1.json (the real log as a drone's first flight)
and health-report-flight5.json (the fifth flight of a synthesized history).
Default out_dir is the Flutter app's assets/docs/samples/.

The 5-flight history reuses the real log's features. Flights 2-5 add seeded
Gaussian noise of sd = min(0.5 * sigma_floor, 5% of |value|) per numeric
feature (1% where a feature has no floor) -- smaller than the spec §2.1 floors, so it's plausible
flight-to-flight scatter, not a fault. Flight 5 additionally carries one
deliberate, disclosed change: vibration p50/p95 x 1.28 (as after a chipped
prop), so the personal-baseline finding shape is visible.
"""

import copy
import datetime
import json
import os
import sys

import numpy as np

from app.services.health import baseline as B
from app.services.health import epochs as E
from app.services.health import norms as N
from app.services.health import report as R
from app.services.health import trends as TR
from app.services.health.features import extract_features
from app.services.health.windowing import window_flight
from app.services.reality_gap.parsers import parse_log

ROOT = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "reality_gap", "real", "px4_777ed799.ulg")
DEFAULT_OUT = os.path.expanduser("~/Desktop/drone_twin_app/assets/docs/samples")
SEED = 20261025
NOISE_FRACTION_OF_FLOOR = 0.5
NOISE_MAX_RELATIVE = 0.05
NO_FLOOR_RELATIVE = 0.01
FLIGHT5_VIBRATION_FACTOR = 1.28
TREND_CREEP = 0.4
INTEGER_FEATURES = {"gps.nsats_p10", "gps.noise", "gps.jamming"}


def vary(features, rng):
    out = copy.deepcopy(features)
    for f in out:
        if f.value is None or f.key in ("act.mean",) or f.key == "vib.clip_rate" and f.value == 0:
            continue
        floor = B.SIGMA_FLOOR.get(f.key)
        if floor is None:      # no spread floor (min cell voltage): 1% scatter
            sd = NO_FLOOR_RELATIVE * abs(f.value)
        else:
            sd = min(NOISE_FRACTION_OF_FLOOR * floor, NOISE_MAX_RELATIVE * abs(f.value))
        f.value = float(f.value + rng.normal(0.0, sd))
        if f.key in INTEGER_FEATURES:
            f.value = float(round(f.value))
    return out


SOURCE_URL = "https://logs.px4.io/plot_app?log=777ed799-cab6-4baa-8f32-80333efc68d8"
APP_ROOT = os.path.expanduser("~/Desktop/drone_twin_app")
BUNDLED_SAMPLE = os.path.join(APP_ROOT, "assets", "data", "sample_flight_report.json")


def sample_block(purpose, provenance, size_bytes, sha256, synthetic=False):
    return {
        "is_sample_data": True,
        "purpose": purpose,
        "provenance": provenance,
        "synthetic": synthetic,
        "source_log": {
            "filename": os.path.basename(FIXTURE), "sha256": sha256, "size_bytes": size_bytes,
            "origin": f"PX4 Flight Review public log, {SOURCE_URL}",
            "licence": "CC-BY (PX4), attributed to PX4 Flight Review",
            "aircraft": "12S hexacopter (CA_ROTOR_COUNT 6, BAT1_N_CELLS 12) on a CubePilot Cube Orange+, "
                        "PX4 1.14.0 dev build (branch mbc_1.14.3), default SDLOG_PROFILE",
        },
        "generated": datetime.date.today().isoformat(),
        "generator": "Drone-Digital-Twin/generate_health_samples.py",
        "seed": SEED if synthetic else None,
    }


REAL_PROVENANCE = ("A real analysis of someone else's flight, not yours: a public PX4 Flight Review log of a "
                   "12S hexacopter, run through this app's health pipeline as if it were a drone's first flight. "
                   "Every figure is computed from that log.")


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)
    parsed = parse_log(FIXTURE)
    window = window_flight(parsed)
    feats = extract_features(parsed, window)
    autopilot = N.autopilot_of(parsed.meta.source_format)
    watched = E.watched_params(parsed.params, parsed.meta.firmware_version)
    size = os.path.getsize(FIXTURE)
    meta = {"filename": parsed.meta.filename, "sha256": parsed.meta.sha256, "source_format": parsed.meta.source_format,
            "size_bytes": size, "duration_s": round(parsed.meta.duration_s, 2),
            "hardware": parsed.meta.hardware, "firmware_release": parsed.meta.firmware_release,
            "firmware_version": parsed.meta.firmware_version, "airframe": parsed.meta.airframe,
            "start_time_utc": parsed.meta.start_time_utc}

    def flight(fid, features, log_meta):
        return R.FlightInput(fid, autopilot, features, window, pack_id=1, watched=dict(watched), log_meta=log_meta,
                             warnings=list(parsed.warnings))

    first = R.analyse([flight(1, feats, meta)])[0]
    first["_sample"] = sample_block("Bundled so the app can show a real report before the student uploads a log.",
                                    REAL_PROVENANCE, size, parsed.meta.sha256)
    rng = np.random.default_rng(SEED)
    history = [flight(1, feats, meta)]
    for fid in range(2, 6):
        fv = vary(feats, rng)
        if fid == 5:
            for f in fv:
                if f.key in ("vib.metric_p50", "vib.metric_p95") and f.value is not None:
                    f.value *= FLIGHT5_VIBRATION_FACTOR
        history.append(flight(fid, fv, dict(meta, filename=f"synthetic_flight_{fid}_from_{parsed.meta.filename}",
                                              sha256=None, size_bytes=None, synthetic=True)))
    fifth = R.analyse(history)[-1]
    fifth["_sample"] = sample_block(
        "Developer fixture showing the personal-baseline report shape. Not bundled in the app.",
        "Flight 5 of a SYNTHESIZED 5-flight history, not real flights. Flight 1 is the real log above; flights "
        f"2-5 reuse its features with seeded noise of sd = min({NOISE_FRACTION_OF_FLOOR} x sigma_floor, "
        f"{int(NOISE_MAX_RELATIVE * 100)}% of value), and flight 5 has vibration p50/p95 multiplied by "
        f"{FLIGHT5_VIBRATION_FACTOR} on purpose.", size, parsed.meta.sha256, synthetic=True)

    # Trends fixture (spec §4.4): 14 synthesized flights on two packs; a
    # declared prop change at flight 7 drops vibration 0.6 m/s^2, after which
    # it creeps up 0.35 m/s^2 per flight; flight 4 has 7 clipping events.
    from app.services.health import trends as TR
    rng_t = np.random.default_rng(SEED + 1)
    base_day = datetime.date(2026, 9, 1)
    t_hist = []
    for i in range(14):
        fv = feats if i == 0 else vary(feats, rng_t)
        if i >= 6:
            fv = copy.deepcopy(fv)
            for f in fv:
                if f.key == "vib.metric_p50" and f.value is not None:
                    f.value = f.value - 0.6 + TREND_CREEP * (i - 6)
        if i == 3:
            fv = copy.deepcopy(fv)
            for f in fv:
                if f.key == "vib.clip_rate":
                    f.value, f.detail = 7 / (window.airborne_s / 60), dict(f.detail, clip_events=7)
        fl = flight(i + 1, fv, dict(meta, synthetic=i > 0))
        fl.pack_id, fl.when = 1 + i % 2, (base_day + datetime.timedelta(days=3 * i)).isoformat()
        t_hist.append(fl)
    trends = TR.build(t_hist, [E.ConfigEvent(6, "props", "user", note="new props")])
    trends["_sample"] = sample_block(
        "Developer fixture for the Trends screen. Not bundled in the app.",
        "14 SYNTHESIZED flights: flight 1 is the real log above, flights 2-14 reuse its features with seeded "
        "noise. A prop change is declared at flight 7, where vibration drops 0.6 m/s², then creeps up "
        f"{TREND_CREEP} m/s² per flight; flight 4 has 7 clipping events; packs alternate 1 and 2.",
        size, parsed.meta.sha256, synthetic=True)

    outputs = [(os.path.join(out_dir, "health-report-flight1.json"), first),
               (os.path.join(out_dir, "trends-sample.json"), trends),
               (os.path.join(out_dir, "health-report-flight5.json"), fifth)]
    if os.path.isdir(os.path.dirname(BUNDLED_SAMPLE)):
        outputs.append((BUNDLED_SAMPLE, first))    # spec §9 step 11: the bundled sample is the computed flight 1
    for path, rep in outputs:
        rep = {"_sample": rep["_sample"], **{k: v for k, v in rep.items() if k != "_sample"}}
        with open(path, "w") as fh:
            json.dump(rep, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Test-log reports and Learn's worked examples
# ---------------------------------------------------------------------------
#   python3 generate_health_samples.py --test-logs ~/Desktop/drone-test-logs
# 1. Runs every log made by make_fixtures.py --health-logs through the real
#    pipeline, grouped and ordered as its README says (one drone per set; each
#    B fault on its own against A1-A5), and saves each report to reports/.
# 2. Writes the app's assets/data/worked_examples.json: the fault reports and
#    the Drift/Step Trends, each with the chart data behind "what it's
#    comparing", computed with the same baseline code the report uses.

WORKED_EXAMPLES = os.path.join(APP_ROOT, "assets", "data", "worked_examples.json")
DEMO_BANNER = "Worked example: a demo, not your flight"
DEMO_PROVENANCE = ("Made for learning. A real 12S hexacopter flight was copied into five normal flights, "
                   "then {change}. Not a real fault, not your drone, and never added to your flights.")

# (id, file, title, glyph, finding to highlight, what the demo changed, what a builder would check)
FAULTS = [
    ("vibration", "B1_vibration-high.ulg", "Vibration high", "vibration", "vib.metric_p50",
     "one more with its vibration raised 1.6x",
     "Check each prop for chips and balance, make sure the prop nuts are tight, and look at the flight controller's mount."),
    ("motor", "B2_motor4-high.ulg", "One motor working harder", "motor", "act.spread",
     "one more with motor 4's output raised by 5%",
     "Spin motor 4 by hand for roughness, check its prop is the right one and undamaged, and look for a bent arm."),
    ("cg", "B3_motor-pair-high.ulg", "Centre of gravity shifted", "balance", "act.spread",
     "one more with the rear pair of motors working 3.5% harder and the front pair 3.5% less",
     "Check where the battery and payload sit; moving the battery forward evens out a rear-heavy drone."),
    ("pack", "B4_pack-resistance-up.ulg", "Pack resistance up", "battery", "pwr.r_cell",
     "one more with the pack's internal resistance raised 1.4x",
     "Watch this pack over the next few flights and compare it with your other packs; check its connector and leads for heat."),
    ("gps", "B5_gps-poor.ulg", "Poor GPS", "gps", "gps.eph_p50",
     "one more with GPS accuracy, satellites and speed accuracy degraded below PX4's requirements",
     "Fly with a clear view of the sky, and move the GPS mast away from power wiring and the video transmitter."),
    ("saturated", "B6_motor-saturated.ulg", "A motor maxed out", "speed", "act.spread",
     "one more with motor 2 needing 1.65x the output, pinning it at full power on hard moves",
     "Check motor 2 and its prop, and whether the drone carries more than its motors can lift with margin."),
    ("estimator", "B7_estimator-rejecting.ulg", "Estimator rejecting a sensor", "estimator", "est.test_ratio_p95",
     "one more with the height test ratio pushed past 1 for 90 s, as a barometer disturbance would",
     "Look at the barometer: foam over it, and shielding from prop wash and sunlight, are the usual fixes."),
]
TRENDS = [
    ("drift", "C", "Vibration creeping up", "trending_up", "vib.metric_p50", "drift",
     "a vibration rise of 3% of its usual on each flight from the sixth",
     "A slow climb is worth catching early: look over the props and motor bearings before it becomes a jump."),
    ("step", "D", "A sudden step", "stairs", "vib.metric_p50", "step",
     "three flights with vibration 1.5x, as if a damaged prop was left on",
     "Something changed between flights 5 and 6: check what was swapped, bumped or crashed in between."),
]


def _plain(o):
    """json.dump fallback: NumPy scalars (np.bool_ from comparisons on float64)
    become plain Python values; anything else is a bug, not a string."""
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"not JSON-serialisable: {type(o).__name__}")


def _test_log_inputs(logs_dir):
    from app.services.reality_gap.parsers import LogParseError
    cache = {}

    def fi(fid, name):
        if name not in cache:
            p = parse_log(os.path.join(logs_dir, name))
            w = window_flight(p)
            m = p.meta
            cache[name] = (N.autopilot_of(m.source_format), extract_features(p, w), w,
                           {"filename": name, "sha256": m.sha256, "source_format": m.source_format,
                            "hardware": m.hardware, "firmware_release": m.firmware_release,
                            "duration_s": round(m.duration_s, 2), "firmware_version": m.firmware_version,
                            "airframe": m.airframe, "start_time_utc": m.start_time_utc},
                           E.watched_params(p.params, m.firmware_version))
        ap, feats, w, meta, watched = cache[name]
        return R.FlightInput(fid, ap, feats, w, watched=dict(watched), log_meta=meta, when=meta["start_time_utc"])
    return fi, LogParseError


def _chart(history, key, current_value):
    """The reference band from the first five flights (median +- 2 x spread),
    the same five behind the finding's "your usual", in display units."""
    from app.services.health import findings as F
    obs = [B.Observation(fl.flight_id, next(f.value for f in fl.features if f.key == key), admitted=True)
           for fl in history[:B.REFERENCE_FLIGHTS]]
    ref = B.reference_baseline(obs, key)
    d = F.DISPLAY[key]
    k = d.scale
    up = B.DIRECTION[key] != B.LOWER_IS_WORSE
    return {"feature": key, "label": d.name, "unit": d.unit, "decimals": d.decimals,
            "baseline_values": [round(o.value * k, 6) for o in obs],
            "value": round(current_value * k, 6) if current_value is not None else None,
            "band": {"low": round((ref.median - TR.BAND_SPREADS * ref.spread) * k, 6),
                     "median": round(ref.median * k, 6),
                     "high": round((ref.median + TR.BAND_SPREADS * ref.spread) * k, 6)},
            "higher_is_worse": up,
            # The band formula can reach below zero; these can't, so the chart starts at 0.
            "nonnegative": key in NONNEGATIVE,
            "band_source": "first five flights: median ± 2 × spread"}


NONNEGATIVE = {"vib.metric_p50", "act.spread", "pwr.r_cell", "gps.eph_p50", "est.test_ratio_p95", "act.sat_frac"}
# What each estimator test ratio compares (PX4 EKF2 innovation test ratios).
ESTIMATOR_CHECK = {"hgt": "the height check, which compares the barometer with the estimate",
                   "mag": "the heading check, which compares the magnetometer with the estimate",
                   "vel": "the velocity check, which compares GPS speed with the estimate",
                   "pos": "the position check, which compares GPS position with the estimate"}


def _band_text(key, band, nonnegative):
    if nonnegative and band["low"] < 0:
        return f"up to {_figure(key, band['high'])}"
    return f"{_figure(key, band['low'])} to {_figure(key, band['high'])}"


def _figure(key, v):
    from app.services.health import findings as F
    return F.figure(v / F.DISPLAY[key].scale, F.DISPLAY[key])


def make_test_log_reports(logs_dir):
    from app.services.health import findings as F
    fi, LogParseError = _test_log_inputs(logs_dir)
    out_dir = os.path.join(logs_dir, "reports")
    os.makedirs(out_dir, exist_ok=True)

    def save(name, obj):
        with open(os.path.join(out_dir, name), "w") as fh:
            json.dump(obj, fh, indent=1, ensure_ascii=False, default=_plain)

    a = [fi(i + 1, f"A{i + 1}_normal.ulg") for i in range(5)]
    for name, rep in zip([f"A{i + 1}_normal.ulg" for i in range(5)], R.analyse(a)):
        save(name + ".report.json", rep)
    names = sorted(os.listdir(logs_dir))
    for b in [n for n in names if n.startswith("B")]:
        save(b + ".report.json", R.analyse(a + [fi(6, b)])[5])
    for prefix in ("C", "D"):
        files = [n for n in names if n.startswith(prefix) and n.endswith(".ulg")]
        hist = [fi(i + 1, n) for i, n in enumerate(files)]
        for n, rep in zip(files, R.analyse(hist)):
            save(n + ".report.json", rep)
        save(f"{prefix}_trends.json", TR.build(hist))
    e = [fi(i + 1, n) for i, n in enumerate(["E3_always-bad.ulg", "E1_too-short.ulg", "E2_no-steady.ulg"])]
    for n, rep in zip(["E3_always-bad.ulg", "E1_too-short.ulg", "E2_no-steady.ulg"], R.analyse(e)):
        save(n + ".report.json", rep)
    try:
        fi(1, "E4_corrupt.ulg")
        save("E4_corrupt.ulg.report.json", {"status": "unexpectedly parsed"})
    except LogParseError as exc:
        save("E4_corrupt.ulg.report.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    save("E5_ardupilot.bin.report.json", R.analyse([fi(1, "E5_ardupilot.bin")])[0])
    print(f"wrote {len(os.listdir(out_dir))} reports to {out_dir}")

    # ---- worked examples
    examples = []
    for ex_id, fname, title, glyph, key, change, real_world in FAULTS:
        rep = R.analyse(a + [fi(6, fname)])[5]
        rep["_sample"] = {"is_sample_data": True, "synthetic": True, "banner_title": DEMO_BANNER,
                          "purpose": "Worked example in Learn. Not bundled as a flight, never cached.",
                          "provenance": DEMO_PROVENANCE.format(change=change),
                          "generated": datetime.date.today().isoformat(),
                          "generator": "Drone-Digital-Twin/generate_health_samples.py --test-logs"}
        f = next((x for x in rep["findings"] if x["key"] == key), None)
        if f is None:
            raise SystemExit(f"{ex_id}: no {key} finding, so no convincing example")
        chart = _chart(a, key, f["value"])
        feat = next(x for x in rep["features"] if x["key"] == key)
        d = F.DISPLAY[key]
        spread = _figure(key, feat["baseline"]["spread"] * d.scale)
        cause = f["causes"][0]["phrase"].rstrip(".") + "." if f["causes"] else ""
        if key == "est.test_ratio_p95":
            cause = f"Worst on {ESTIMATOR_CHECK[feat['detail']['worst']]}."
        lower = d.name if d.name[:3].isupper() else d.name[0].lower() + d.name[1:]
        comparing = (f"The band is this drone's usual {lower} over its first five flights: "
                     f"{_band_text(key, chart['band'], chart['nonnegative'])}. This flight: {_figure(key, chart['value'])}.")
        # The spread is max(1.4826 x MAD, the feature's floor): what the app
        # allows as normal wobble, not necessarily what was measured.
        why = (f"The app allows about {spread} of normal flight-to-flight wobble; this flight is "
               f"{round(feat['z'])} times that above its usual. {cause}")
        sat = next((x for x in rep["findings"] if x["key"] == "act.sat_frac"), None)
        if ex_id == "saturated" and sat:
            # The point of this example is the full output; the general cause is on the finding card.
            why = why[: why.index("Most often")] + f"It hit full output {sat['figure']} of the flight: no margin left."
        why = F.check_claim(why)
        examples.append({"id": ex_id, "kind": "report", "title": title, "glyph": glyph, "symptom": f["line"],
                         "focus": key, "figure": f["figure"],
                         "seeing": F.check_claim("The highlighted finding is what the app flags on this flight."),
                         "comparing": F.check_claim(comparing), "why": why, "real_world": F.check_claim(real_world),
                         "chart": chart, "report": rep})
    for ex_id, prefix, title, glyph, key, kind, change, real_world in TRENDS:
        files = [n for n in sorted(os.listdir(logs_dir)) if n.startswith(prefix) and n.endswith(".ulg")]
        hist = [fi(i + 1, n) for i, n in enumerate(files)]
        tr = TR.build(hist)
        tr["_sample"] = {"is_sample_data": True, "synthetic": True, "banner_title": DEMO_BANNER,
                         "provenance": DEMO_PROVENANCE.format(change=change).replace(
                             "five normal flights, then", f"{len(files)} flights: five normal, then")}
        series = next(x for c in tr["cards"] for x in c["series"] if x["feature"] == key)
        call = next((c for c in series["callouts"] if c["kind"] == kind), None)
        if call is None:
            raise SystemExit(f"{ex_id}: no {kind} callout, so no convincing example")
        chart = _chart(hist, key, None)
        chart["later_values"] = [round(p["value"] * F.DISPLAY[key].scale, 6) for p in series["points"][5:]]
        b = chart["band"]
        comparing = (f"The band is the first five flights: {_band_text(key, b, chart['nonnegative'])}. "
                     f"Each dot after it is one more flight.")
        why = ("No single flight jumped, so each report stayed calm. The median of the last five left the band."
               if kind == "drift" else
               "Three flights in a row landed well outside the band, straight after a normal one.")
        examples.append({"id": ex_id, "kind": "trends", "title": title, "glyph": glyph, "symptom": call["text"],
                         "focus": key, "callout": kind,
                         "seeing": F.check_claim(f"The highlighted {kind} callout is what Trends flags across these flights."),
                         "comparing": F.check_claim(comparing), "why": F.check_claim(why),
                         "real_world": F.check_claim(real_world), "chart": chart, "trends": tr})
    with open(WORKED_EXAMPLES, "w") as fh:
        json.dump({"_generated": {"by": "Drone-Digital-Twin/generate_health_samples.py --test-logs",
                                  "date": datetime.date.today().isoformat(),
                                  "logs": "make_fixtures.py --health-logs (perturbed copies of real/px4_777ed799.ulg)"},
                   "examples": examples}, fh, ensure_ascii=False, separators=(",", ":"), default=_plain)
    print(f"wrote {WORKED_EXAMPLES} ({os.path.getsize(WORKED_EXAMPLES) / 1024:.0f} KB, {len(examples)} examples)")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--test-logs":
        make_test_log_reports(os.path.expanduser(sys.argv[2]))
    else:
        main()
