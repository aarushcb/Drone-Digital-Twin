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

    outputs = [(os.path.join(out_dir, "health-report-flight1.json"), first),
               (os.path.join(out_dir, "health-report-flight5.json"), fifth)]
    if os.path.isdir(os.path.dirname(BUNDLED_SAMPLE)):
        outputs.append((BUNDLED_SAMPLE, first))    # spec §9 step 11: the bundled sample is the computed flight 1
    for path, rep in outputs:
        rep = {"_sample": rep["_sample"], **{k: v for k, v in rep.items() if k != "_sample"}}
        with open(path, "w") as fh:
            json.dump(rep, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
