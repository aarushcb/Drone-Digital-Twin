"""
Trends -- health spec §4.4; build step 13.

Raw feature values in their own units, one point per qualifying flight,
against the epoch's FROZEN reference (median ± 2 × spread). Not scores: a
score is already "versus your recent normal", so plotting it hides the slow
drift Trends exists to catch.

Callouts, all in feature units:
  step   the last 3 flights in the epoch are all outside the reference band
         on the harmful side                       (needs the reference)
  drift  the median of the last 5 flights in the epoch is outside the band
         on the harmful side                       (needs the reference)
  guide  the last 3 flights are all past a published guide line
                                                   (any n, labelled "vs ... guide")

A declared or detected change starts a new epoch for the features it
resets: earlier points are greyed, the band restarts, and the step itself
never raises a callout because nothing after it is compared with before.
"""

import statistics
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from . import baseline as B
from . import epochs as E
from . import findings as F
from . import norms as N
from . import report as R
from .features import ATT, POS

STEP_FLIGHTS = 3          # spec §4.4
DRIFT_FLIGHTS = 5         # spec §4.4
GUIDE_FLIGHTS = 3         # spec §4.4
BAND_SPREADS = 2.0        # reference band = median ± 2 × spread
MIN_SPAN_FLOORS = 4.0     # vertical scale at least 4 × sigma_floor wide

CARDS = (
    ("vibration", "Vibration", ("vib.metric_p50", "vib.imbalance")),
    ("motors", "Motors", ("act.spread", "act.hover")),
    ("power", "Power", ("pwr.r_cell",)),
)
NOT_TRENDED = (
    {"subsystem": ATT, "title": "Attitude & control",
     "text": "Not trended: depends on wind and how hard it was flown more than on the drone"},
    {"subsystem": POS, "title": "Position",
     "text": "Not trended: depends on the flying site (sky view, interference) more than on the drone"},
)
CAVEATS = {"act.hover": "Also moved by payload, temperature and altitude, not only by wear."}

CHANGE_LABELS = {
    "props": ("New props", "the new props"), "motors_escs": ("New motors/ESCs", "the new motors or ESCs"),
    "pack": ("Battery change", "the battery change"), "frame": ("Frame change", "the frame change"),
    "payload": ("Payload change", "the payload change"),
    "firmware_tuning": ("Firmware/tuning change", "the firmware or tuning change"),
    "gps": ("GPS change", "the GPS change"), "inferred_confirmed": ("Confirmed change", "the change you confirmed"),
}


def _date_label(when: Optional[str]) -> Optional[str]:
    try:
        d = datetime.fromisoformat(str(when))
    except (TypeError, ValueError):
        return None
    return None if d.year < 2000 else f"{d.day} {d.strftime('%b')}"


def _num(key: str, v: float) -> str:
    """A bare number, except percentages keep their sign: "50%" reads, "50" doesn't."""
    n = F._num(v, F.DISPLAY[key])
    return n + "%" if _unit(key) == "%" else n


def _unit(key: str) -> str:
    return F.DISPLAY[key].unit


def _with_unit(key: str, text: str) -> str:
    u = _unit(key)
    if u == "%":
        return text                     # _num already carries it
    return text + u if u == "°" else (f"{text} {u}" if u else text)


def _guides(key: str, autopilot: str) -> List[dict]:
    norm = N.norm_for(key, autopilot)
    if norm is None or norm.kind not in (N.GRADED, N.BAD_ONLY) or norm.uses_detail:
        return []
    owner = "ArduPilot" if "ardupilot.org" in norm.url else "PX4"
    out = []
    if norm.kind == N.GRADED and norm.healthy_edge:
        out.append({"value": norm.healthy_edge, "kind": "healthy_edge", "label": f"{owner} guide", "source": norm.source})
    if norm.bad_edge is not None:
        out.append({"value": norm.bad_edge, "kind": "bad_edge", "label": f"{owner} guide", "source": norm.source})
    return out


def _harmful(key: str, v: float, edge: float) -> bool:
    return v > edge if B.DIRECTION[key] != B.LOWER_IS_WORSE else v < edge


def build(flights: Sequence[R.FlightInput], declared: Sequence[E.ConfigEvent] = ()) -> dict:
    flights = list(flights)
    reports = R.analyse(flights, declared)
    events = R.detect_events(flights, declared)
    feat = [{f["key"]: f for f in rep["features"]} for rep in reports]
    autopilot = flights[-1].autopilot if flights else "px4"
    cards = []
    for card_key, title, keys in CARDS:
        series = []
        for key in keys:
            packs = sorted({fl.pack_id for fl in flights}, key=lambda p: (p is None, p)) if key.startswith("pwr.") else [None]
            for pack in packs:
                s = _series(key, pack, key.startswith("pwr."), flights, reports, feat, events, autopilot)
                if s is not None:
                    series.append(s)
        markers = []
        if card_key == "vibration":
            for i, fl in enumerate(flights):
                ev = (feat[i].get("vib.clip_rate") or {}).get("detail", {}).get("clip_events")
                if fl.window.qualified and ev:
                    markers.append({"flight_number": i + 1, "kind": "clipping", "events": int(ev),
                                    "text": F.check_claim(f"{int(ev)} accelerometer clipping events")})
        cards.append({"key": card_key, "title": title, "series": series, "markers": markers,
                      "callouts": [c for s in series for c in s["callouts"]]})
    out = {"flights": sum(1 for fl in flights if fl.window.qualified), "cards": cards, "not_trended": list(NOT_TRENDED)}
    F.assert_report_honest(out)
    return out


def _series(key, pack, per_pack, flights, reports, feat, events, autopilot) -> Optional[dict]:
    pts = []
    for i, fl in enumerate(flights):
        f = feat[i].get(key)
        if not fl.window.qualified or f is None or f["value"] is None or (per_pack and fl.pack_id != pack):
            continue
        pts.append({"flight_id": fl.flight_id, "flight_number": i + 1, "date": fl.when, "value": f["value"],
                    "figure": f["figure"], "group": f["group"], "admitted": f["admitted"], "score": f["score"],
                    "_detail": f.get("detail") or {}})
    if not pts:
        return None
    current_group = pts[-1]["group"]
    for p in pts:
        p["current"] = p["group"] == current_group
    current = [p for p in pts if p["current"]]
    admitted = [B.Observation(p["flight_id"], p["value"]) for p in current if p["admitted"]]
    ref = B.reference_baseline(admitted, key)
    name = F.DISPLAY[key].name
    band = None
    if ref is not None:
        band = {"median": round(ref.median, 6), "spread": round(ref.spread, 6),
                "low": round(ref.median - BAND_SPREADS * ref.spread, 6),
                "high": round(ref.median + BAND_SPREADS * ref.spread, 6), "n": ref.n, "flight_ids": ref.flight_ids}
    guides = _guides(key, autopilot)

    # Boundaries: changes that reset this feature, inside the plotted range.
    first, last = pts[0]["flight_number"] - 1, pts[-1]["flight_number"] - 1
    boundaries, start_event = [], None
    for ev in events:
        if key in ev.reset_features and first < ev.flight_index <= last:
            label, phrase = CHANGE_LABELS.get(ev.change_type, (ev.change_type, "the change"))
            date = _date_label(flights[ev.flight_index].when)
            boundaries.append({"flight_number": ev.flight_index + 1, "change_type": ev.change_type, "source": ev.source,
                               "label": f"{label}, {date}" if date else label})
            if ev.flight_index + 1 <= current[0]["flight_number"]:
                start_event = phrase

    callouts = []
    vals = [p["value"] for p in current]
    if band is not None:
        up = B.DIRECTION[key] != B.LOWER_IS_WORSE
        edge = band["high"] if up else band["low"]
        word = "up" if up else "down"
        if len(vals) >= STEP_FLIGHTS and all(_harmful(key, v, edge) for v in vals[-STEP_FLIGHTS:]):
            a, b, c = (_num(key, v) for v in vals[-STEP_FLIGHTS:])
            callouts.append({"kind": "step", "feature": key, "pack_id": pack, "text": F.check_claim(
                f"{name} stepped {word}: {a}, {b} and {_with_unit(key, c)} on the last 3 flights, outside your "
                f"usual {_num(key, band['low'])}–{_with_unit(key, _num(key, band['high']))}")})
        if len(vals) >= DRIFT_FLIGHTS:
            med = statistics.median(vals[-DRIFT_FLIGHTS:])
            if _harmful(key, med, edge):
                since = f" since {start_event}" if start_event else ""
                callouts.append({"kind": "drift", "feature": key, "pack_id": pack, "text": F.check_claim(
                    f"{name} has crept {word} from {_num(key, ref.median)} to {_with_unit(key, _num(key, med))} "
                    f"over {len(vals)} flights{since}")})
    if guides and len(vals) >= GUIDE_FLIGHTS:
        g = min(guides, key=lambda x: x["value"]) if B.DIRECTION[key] != B.LOWER_IS_WORSE else max(guides, key=lambda x: x["value"])
        if all(_harmful(key, v, g["value"]) for v in vals[-GUIDE_FLIGHTS:]):
            callouts.append({"kind": "guide", "feature": key, "pack_id": pack, "label": f"vs {g['label']}",
                             "text": F.check_claim(f"{name} past {g['label'].replace(' guide', '')}'s "
                                                   f"{g['value']:.1f} guide on the last 3 flights")})

    latest = current[-1]
    delta = None
    if band is not None:
        d = latest["value"] - ref.median
        shown = F.figure(abs(d), F.DISPLAY[key])
        sign = "±" if float(F._num(abs(d), F.DISPLAY[key])) == 0 else ("+" if d > 0 else "−")
        delta = {"value": round(d, 6), "figure": sign + shown,
                 "score": latest["score"], "vs": "reference"}
    floor = max(B.SIGMA_FLOOR.get(key, 0.0),
                B.RELATIVE_FLOOR.get(key, 0.0) * abs(ref.median if ref is not None else latest["value"]))
    annotation = None
    if key == "act.spread" and latest["_detail"].get("highest_motor_number"):
        annotation = f"highest: motor {latest['_detail']['highest_motor_number']}"
    for p in pts:
        p.pop("_detail")
    return {
        "feature": key, "name": name, "unit": _unit(key), "scale": F.DISPLAY[key].scale, "pack_id": pack,
        "label": name + (f" · pack {pack}" if per_pack and pack is not None else (" · pack unknown" if per_pack else "")),
        "direction": B.DIRECTION[key], "points": pts, "reference": band,
        "reference_progress": None if band is not None else {"have": len(admitted), "need": B.REFERENCE_FLIGHTS},
        "guides": guides, "boundaries": boundaries, "callouts": callouts, "latest_delta": delta,
        "min_span": round(MIN_SPAN_FLOORS * floor, 6), "caveat": CAVEATS.get(key), "annotation": annotation,
    }
