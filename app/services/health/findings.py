"""
Findings generator -- health spec §5 and §6; build step 9.

A finding is emitted when a feature's score is below 85, on any absolute bad,
or on a definitional fact (a motor at full output, the estimator at PX4's own
rejection level). The worst three go to the verdict.

How §6 ("what we cannot honestly say") is enforced in code, not just prose:
  1. Closed vocabulary. No free text enters a finding: every line comes from
     a template below, every cause from the CAUSES table. There is no
     parameter through which a remaining-life figure, a battery percentage or
     a part verdict could be passed.
  2. Every template and every cause string is checked against the §6 deny
     list at import time, so a forbidden phrase added to this file fails on
     load, before any report exists.
  3. Every emitted string -- line, hedge, cause phrase, verdict, prompt -- is
     checked again at emission (check_claim), and the assembled report is
     walked for forbidden keys and strings (assert_report_honest). A
     violation raises ForbiddenClaim; it is never silently dropped.
  4. Structural rules: a finding needs a finite figure (§5.1 "never invent a
     figure"), and every cause phrase must open with probability wording
     ("Most often", "Also check", "Worth checking"), never a diagnosis.

Limit, stated plainly: a deny list cannot prove the absence of every
paraphrase. Rule 1 is the real guarantee; rules 2-3 catch regressions in
this file and in anything that later edits a report.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .features import ATT, MOT, POS, PWR, VIB
from .scoring import FeatureScore

MAX_LINE_WORDS = 9                       # spec §5.1: <= 9 words before the hedge
VERDICT_MAX_FINDINGS = 3                 # spec §5.1
EMIT_BELOW = 85.0                        # spec §5.1
DEFINITIONAL_SORT_SCORE = 70.0           # definitional facts carry no score; ranked with "watch"


class ForbiddenClaim(ValueError):
    """Raised when text or a report would say something spec §6 forbids."""


# ---------------------------------------------------------------- spec §6 deny list

FORBIDDEN = [
    (r"remaining life|life (left|remaining)|lifespan|life expectancy|until (it )?fail|wear out in|"
     r"\b\d+\s*(more )?(hours?|cycles?|flights?) (left|remaining)\b|end of (its )?life", "§6.1 remaining life"),
    (r"\b(is|has|was) (failed|failing|broken|dead|faulty|defective|bad)\b|\bfailed (motor|esc|prop|arm|cell)\b|"
     r"\bfaulty\b|\bdiagnos|\bthe cause is\b|\bcaused by\b|\bdefinitely\b|\bcertainly\b", "§6.2 specific failed part / diagnosis"),
    (r"state of health|\bsoh\b|\b\d+(\.\d+)?\s*%\s*(health|healthy|capacity|charge|battery|remaining|left)\b|"
     r"remaining capacity|capacity (left|remaining|fade of)|\b(battery|pack) (health|life)\b|"
     r"health\s*(of\s*|at\s*)?\d+(\.\d+)?\s*%", "§6.3 battery % / capacity"),
    (r"blade[- ]?pass|rotor (frequency|speed)|\b\d+(\.\d+)?\s*hz\b", "§6.4 rotor-frequency vibration"),
    (r"\brpm\b|motor (temperature|temp|current)|esc (temperature|temp|current)", "§6.5 per-motor RPM/current/temp"),
    (r"safe to fly|\bunsafe\b|airworth|cleared (for|to) fl|fit to fly|ok to fly|certif", "§6.6 safe to fly"),
    (r"better than (average|most|other)|worse than (average|most|other)|other drones|\bfleet\b|"
     r"\baverage drone\b|than most drones|percentile", "§6.7 comparison with other drones"),
    (r"(due to|because of|caused by) (the )?wind|not (the )?wind|wind,? not|mechanical,? not wind", "§6.8 wind vs mechanical"),
    (r"because you|since you (swapped|changed|replaced|installed)", "§6.9 cause of an unrecorded change"),
    (r"\b(pids?|tune|tuning|gains) (is|are) (bad|good|poor|wrong|off|great)\b|badly tuned|well tuned|poorly tuned|"
     r"\bbad (tune|tuning|pids?)\b", "§6.10 absolute tuning quality"),
    (r"\b\d+(\.\d+)?\s*(g|kg|grams?) (heavier|lighter|of payload)\b|payload (is|weighs|of) \d|heavier by \d",
     "§6.11 payload mass"),
    (r"(same as|equivalent to|equals) (ardupilot|px4)|(ardupilot|px4) equivalent", "§6.12 cross-autopilot equivalence"),
    (r"simulat|sim[- ]vs[- ]real|digital twin|fidelity", "§6.13 simulator fidelity"),
]
_FORBIDDEN_RE = [(re.compile(p, re.IGNORECASE), why) for p, why in FORBIDDEN]

# Report keys that could only exist to carry a §6 claim.
FORBIDDEN_KEYS = {
    "remaining_life", "life_remaining", "hours_remaining", "cycles_remaining", "flights_remaining",
    "soh", "state_of_health", "battery_pct", "battery_percent", "battery_health_pct", "capacity_remaining",
    "remaining_capacity", "failed_part", "failed_component", "diagnosis", "safe_to_fly", "airworthy",
    "fleet_rank", "fleet_percentile", "rpm", "motor_rpm", "motor_temp", "motor_current", "payload_mass_kg",
    "sim_score", "fidelity_score", "rotor_frequency_hz", "blade_pass_hz", "tuning_quality",
}

PROBABILITY_OPENERS = ("Most often", "Also check", "Worth checking")


def check_claim(text: str) -> str:
    for rx, why in _FORBIDDEN_RE:
        m = rx.search(text)
        if m:
            raise ForbiddenClaim(f"{why}: {m.group(0)!r} in {text!r}")
    return text


def assert_report_honest(obj, path: str = "report") -> None:
    """Walk a report: no forbidden key anywhere, no forbidden phrase in any
    user-facing string. Provenance fields (_sample, real_log_meta) are
    descriptions of where data came from, not claims, and are skipped."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in FORBIDDEN_KEYS:
                raise ForbiddenClaim(f"§6 forbidden key {k!r} at {path}")
            if k in ("_sample", "real_log_meta"):
                continue
            assert_report_honest(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            assert_report_honest(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        check_claim(obj)


# ---------------------------------------------------------------- display

@dataclass(frozen=True)
class Display:
    name: str
    unit: str = ""          # appended after a space; "%" attaches
    decimals: int = 1
    scale: float = 1.0      # e.g. 100 for ratios shown as %
    derived: bool = False   # spec §5.1: derived figures say "about"
    short: str = ""         # one-word fallback when the full name can't fit 9 words


DISPLAY: Dict[str, Display] = {
    "vib.metric_p50": Display("Vibration", "m/s²", 1),
    "vib.metric_p95": Display("Peak vibration", "m/s²", 1),
    "vib.clip_rate": Display("Clipping", "/min", 1),
    "vib.band_lo": Display("Low-band vibration", "dB", 1),
    "vib.band_struct": Display("Frame-band vibration", "dB", 1),
    "vib.imbalance": Display("Prop imbalance", "", 2),
    "act.spread": Display("Motor spread", "%", 0, 100.0),
    "act.sat_frac": Display("Motor saturation", "%", 1, 100.0),
    "act.hover": Display("Hover throttle", "%", 0, 100.0),
    "pwr.r_cell": Display("Pack resistance", "mΩ/cell", 1, derived=True, short="Resistance"),
    "pwr.sag_hover": Display("Hover sag", "V/cell", 2),
    "pwr.v_cell_min": Display("Lowest cell", "V/cell", 2),     # usage note only; never a finding
    "trk.rate_rms.roll": Display("Roll tracking error", "%", 0, 100.0),
    "trk.rate_rms.pitch": Display("Pitch tracking error", "%", 0, 100.0),
    "trk.rate_rms.yaw": Display("Yaw tracking error", "%", 0, 100.0),
    "trk.att_err_p95": Display("Attitude error", "°", 1),
    "gps.eph_p50": Display("GPS error", "m", 2),
    "gps.nsats_p10": Display("Satellites", "", 0),
    "gps.sacc_p50": Display("Speed error", "m/s", 2),
    "gps.noise": Display("GPS noise", "", 0),
    "gps.jamming": Display("GPS jamming", "", 0),
    "est.test_ratio_p95": Display("Estimator ratio", "", 2),
}


def _num(v: float, d: Display) -> str:
    return f"{v * d.scale:.{d.decimals}f}"


def figure(v: float, d: Display) -> str:
    n = _num(v, d)
    if d.unit == "%" or d.unit == "°":
        return n + d.unit
    return f"{n} {d.unit}" if d.unit else n


def _words(s: str) -> int:
    return len(s.split())


def _first_fitting(candidates: Sequence[str]) -> str:
    for c in candidates:
        if _words(c) <= MAX_LINE_WORDS:
            return c
    raise ValueError(f"no line template fits {MAX_LINE_WORDS} words: {candidates[-1]!r}")


# ---------------------------------------------------------------- causes (spec §5.3)

# (most often, also check). Rows not in §5.3 are marked; they need spec review.
CAUSES: Dict[str, tuple] = {
    "vib.metric": (["a damaged or unbalanced prop, or a loose prop nut"],
                   ["motor bearing wear", "a hardened or loose flight-controller mount or damper"]),
    "vib.imbalance": (["propeller imbalance, which PX4's detector is designed for"],
                      ["a prop on backwards, or a wrong pitch on one arm"]),
    "vib.clip_rate": (["severe vibration saturating the accelerometer"],
                      ["hard landings: check whether the clipping coincides with touchdown"]),
    "vib.band_struct": (["frame or arm flex, or a loose arm clamp"],
                        ["soft damper resonance", "a loose battery strap"]),
    # Not in §5.3: 0-5 Hz is the band where control oscillation and flying
    # inputs live. Needs spec review.
    "vib.band_lo": (["low-frequency oscillation from tuning or a soft mount"],
                    ["wind gusts or aggressive flying"]),
    "pattern.single_motor_high": (["that motor or prop producing less thrust: a chipped or wrong prop, a weak "
                                   "motor, or its ESC"], ["that arm bent or twisted, tilting its thrust"]),
    "pattern.one_side_high": (["the centre of gravity shifted toward that side (battery or payload placement)"], []),
    "pattern.same_spin_set_high": (["yaw imbalance: a motor mounted tilted, or arms twisted"],
                                   ["a prop of different pitch on that set"]),
    "pattern.other": (["one motor or prop producing less thrust than the rest"],
                      ["an arm bent or twisted", "the centre of gravity shifted"]),
    "act.hover": (["a heavier vehicle, worn props, or a tired pack (lower voltage needs more throttle)"],
                  ["hot weather or high altitude (thinner air)"]),
    "act.sat_frac": (["underpowered or overloaded for the manoeuvre"], ["aggressive flying", "strong wind"]),
    "pwr.r_cell": (["pack ageing"], ["a cold pack", "a worn connector or solder joint"]),
    # Not in §5.3: sag rises with hover current as well as with resistance
    # (spec §1.3 F5). Needs spec review.
    "pwr.sag_hover": (["more current at hover than usual (more throttle)"],
                      ["pack ageing: compare pack resistance", "a cold pack"]),
    "trk": (["gyro noise from vibration (check vibration first), or a change in tuning or props"],
            ["wind gusts", "mechanical play", "motors saturating"]),
    "gps.position": (["poor sky view or interference at the flying site"],
                     ["GPS antenna placement or a loose mount", "power wiring near the GPS"]),
    "gps.rf": (["electromagnetic interference from the drone's own wiring or VTX"], ["a nearby external source"]),
    "est": (["sensors disagreeing, most often the magnetometer near power wiring"],
            ["baro disturbed by prop wash", "GPS glitches"]),
}


def _cause_key(fs: FeatureScore) -> str:
    k = fs.key
    if k.startswith("vib.metric"):
        return "vib.metric"
    if k == "act.spread":
        label = (fs.feature.detail or {}).get("pattern_label")
        return f"pattern.{label}" if f"pattern.{label}" in CAUSES else "pattern.other"
    if k.startswith("trk."):
        return "trk"
    if k in ("gps.eph_p50", "gps.nsats_p10", "gps.sacc_p50"):
        return "gps.position"
    if k in ("gps.noise", "gps.jamming"):
        return "gps.rf"
    if k.startswith("est."):
        return "est"
    return k


def causes_for(fs: FeatureScore, confidence: str) -> List[dict]:
    most, also = CAUSES[_cause_key(fs)]
    out = []
    for rank, (text, likelihood) in enumerate([(t, "most_often") for t in most] + [(t, "also_check") for t in also], 1):
        if confidence == "low":
            opener = "Worth checking"          # spec §5.2: low confidence -> "worth checking", not "likely"
        else:
            opener = "Most often" if likelihood == "most_often" else "Also check"
        out.append({"rank": rank, "likelihood": likelihood, "text": text, "phrase": f"{opener}: {text}"})
    return out


# ---------------------------------------------------------------- lines (spec §5.1)

HEDGES = {"high": "", "medium": "Looks like ", "low": "Early read: "}   # spec §5.2


def hedge(line: str, confidence: str) -> str:
    h = HEDGES[confidence]
    if h == "Looks like " and len(line) > 1 and line[1].islower():
        line = line[0].lower() + line[1:]
    return h + line


def _personal_line(d: Display, fs: FeatureScore) -> str:
    v, med = fs.feature.value, fs.rolling.median
    word = "up" if v >= med else "down"
    fig = ("about " if d.derived else "") + figure(v, d)
    delta = _num(abs(v - med), d)
    usual = _num(med, d) + ("%" if d.unit == "%" else "")
    return _first_fitting([f"{d.name} {fig}, {word} {delta} on your usual {usual}",
                           f"{d.name} {fig}, {word} {delta} vs usual {usual}",
                           f"{d.short or d.name} {fig}, {word} {delta} vs usual {usual}"])


def _absolute_line(d: Display, fs: FeatureScore) -> str:
    norm, v = fs.absolute.norm, fs.feature.value
    owner = "ArduPilot" if "ardupilot.org" in norm.url else "PX4"
    if norm.kind == "gate":
        fig = figure(v, d)
        edge = f"{norm.healthy_edge:g}"
        verb = "below" if norm.direction == "lower_is_worse" else "worse than"
        return _first_fitting([f"{d.name} {fig}, {verb} what PX4 requires to trust GPS",
                               f"{d.name} {fig}, outside PX4's {edge} GPS-trust bar",
                               f"{d.name} {fig}, outside PX4's GPS-trust bar"])
    if norm.kind == "bad_only":
        return _first_fitting([f"{d.name} {figure(v, d)}, past {owner}'s {norm.bad_edge:g} failure trigger"])
    if norm.uses_detail == "clip_events":
        events = int(fs.feature.detail.get("clip_events", 0))
        if events > norm.bad_edge:
            return _first_fitting([f"Clipping {events} events, above {owner}'s {norm.bad_edge:g} guide"])
        return _first_fitting([f"Clipping {events} events, above {owner}'s ideal of zero"])
    edge = norm.bad_edge if v > norm.bad_edge else norm.healthy_edge
    return _first_fitting([f"{d.name} {figure(v, d)}, above {owner}'s {edge:.1f} guide"])


def _consistent_line(d: Display, fs: FeatureScore) -> str:
    side = "low" if fs.absolute.norm and fs.absolute.norm.direction == "lower_is_worse" else "high"
    fig = figure(fs.feature.value, d)
    return _first_fitting([f"{d.name} {fig}, {side} on every flight so far",
                           f"{d.name} {fig}, {side} on every flight"])


def _pattern_line(fs: FeatureScore) -> Optional[str]:
    """Spec §5.1 pattern template. The figure is the high motor (or set)
    over the mean of the rest, from the steady-flight per-motor means."""
    det = fs.feature.detail or {}
    per = det.get("per_motor")
    label = det.get("pattern_label")
    if not per or label not in ("single_motor_high", "one_side_high"):
        return None
    per = [float(x) for x in per]
    if label == "single_motor_high":
        m = int(det.get("pattern_motor_number") or (per.index(max(per)) + 1))
        rest = [x for i, x in enumerate(per) if i != m - 1]
        excess = 100.0 * (per[m - 1] / (sum(rest) / len(rest)) - 1.0)
        return _first_fitting([f"Motor {m} working {excess:.0f}% harder than the others"])
    high = [i - 1 for i in det.get("pattern_high_motors", [])]
    if not high or len(high) == len(per):
        return None
    hi = sum(per[i] for i in high) / len(high)
    lo = sum(x for i, x in enumerate(per) if i not in high) / (len(per) - len(high))
    side = det.get("pattern_toward", "one side")
    return _first_fitting([f"{side.capitalize()} motors working {100.0 * (hi / lo - 1.0):.0f}% harder than the rest"])


def _definitional_line(fs: FeatureScore) -> str:
    d = DISPLAY[fs.key]
    if fs.definitional == "motor_saturation":
        return _first_fitting([f"Motors at full output {figure(fs.feature.value, d)} of the flight"])
    return _first_fitting([f"Estimator ratio {figure(fs.feature.value, d)}, past PX4's rejection level"])


# ---------------------------------------------------------------- findings

@dataclass
class Finding:
    key: str
    subsystem: str
    kind: str                 # personal | absolute | consistently_bad | pattern | definitional
    line: str                 # <= 9 words, no hedge
    text: str                 # hedge + line: what the UI shows
    figure: str
    value: float
    comparison: Optional[dict]
    mode: str
    n: int
    confidence: str
    score: Optional[float]
    causes: List[dict]
    quarantined: bool = False
    basis: Optional[str] = None
    cause_group: str = ""     # the §5.3 cause row: findings sharing one are one finding
    related: List[dict] = field(default_factory=list)   # other features collapsed into this one
    sort_score: float = field(default=0.0, repr=False)
    tier: str = field(default="R1", repr=False)

    def to_json(self) -> dict:
        d = dict(self.__dict__)
        d.pop("sort_score")
        d.pop("tier")
        return d


def _make(fs: FeatureScore, kind: str, line: str, comparison: Optional[dict], confidence: str) -> Finding:
    """The only constructor. Enforces §5.1 (real, finite figure; <= 9 words)
    and §6 (every string checked) before a Finding can exist."""
    v = fs.feature.value
    if v is None or not math.isfinite(float(v)):
        raise ValueError(f"§5.1: no finding without a real value ({fs.key})")
    if _words(line) > MAX_LINE_WORDS:
        raise ValueError(f"§5.1: line over {MAX_LINE_WORDS} words: {line!r}")
    causes = causes_for(fs, confidence)
    for c in causes:
        if not c["phrase"].startswith(PROBABILITY_OPENERS):
            raise ForbiddenClaim(f"§5.2: cause not phrased as a probability: {c['phrase']!r}")
    text = hedge(line, confidence)
    for s in [line, text] + [c["phrase"] for c in causes]:
        check_claim(s)
    d = DISPLAY[fs.key]
    sort = fs.score if fs.score is not None else DEFINITIONAL_SORT_SCORE
    if fs.absolute_bad:
        sort = min(sort, -1.0 + sort / 100.0)  # absolute bads always first
    return Finding(fs.key, fs.feature.subsystem, kind, line, text, figure(v, d), float(v), comparison, fs.mode,
                   fs.n, confidence, fs.score, causes, quarantined=fs.admitted is False and fs.quarantine_reason
                   not in (None, "flight_not_qualified", "feature_invalid"), basis=fs.basis,
                   cause_group=_cause_key(fs), sort_score=sort, tier=fs.feature.tier)


def finding_for(fs: FeatureScore) -> Optional[Finding]:
    if fs.key not in DISPLAY or fs.feature.value is None:
        return None
    emit = (fs.status == "scored" and fs.score is not None and fs.score < EMIT_BELOW) or fs.absolute_bad
    if not emit:
        if fs.definitional:
            return _make(fs, "definitional", _definitional_line(fs),
                         {"kind": "definition", "edge": 0.0 if fs.key == "act.sat_frac" else 1.0},
                         fs.confidence or "high")
        return None
    d = DISPLAY[fs.key]
    conf = fs.confidence or "low"
    norm = fs.absolute.norm
    if fs.consistently_bad:
        return _make(fs, "consistently_bad", _consistent_line(d, fs),
                     {"kind": "guide", "edge": norm.bad_edge if norm.bad_edge is not None else norm.healthy_edge,
                      "flights": fs.flights_in_group}, conf)
    personal = fs.rolling is not None and fs.weight > 0.5 or (fs.absolute.score is None and fs.rolling is not None)
    if fs.key == "act.spread" and personal:
        line = _pattern_line(fs)
        if line:
            return _make(fs, "pattern", line, {"kind": "baseline", "median": round(fs.rolling.median, 4),
                                               "n": fs.rolling.n, "group": fs.group}, conf)
    if personal:
        return _make(fs, "personal", _personal_line(d, fs),
                     {"kind": "baseline", "median": round(fs.rolling.median, 4), "spread": round(fs.rolling.spread, 4),
                      "n": fs.rolling.n, "z": round(fs.z, 2), "group": fs.group}, conf)
    edge = norm.bad_edge if norm.kind == "bad_only" else (
        norm.bad_edge if norm.bad_edge is not None and fs.feature.value > norm.bad_edge else norm.healthy_edge)
    return _make(fs, "absolute", _absolute_line(d, fs),
                 {"kind": "guide", "edge": edge, "norm_kind": norm.kind, "source": norm.source, "url": norm.url,
                  "checked": norm.checked}, conf)


def generate(scores: Sequence[FeatureScore]) -> List[Finding]:
    """
    One finding per probable cause, not per feature (spec §5.1 grouping
    rule). Findings whose causes come from the same §5.3 row -- vibration
    p50 and p95, the three tracking axes, the three GPS position figures --
    describe one physical problem, so they collapse into one. Its line is
    the most telling member: the worst score (absolute bads first), then
    tier R1 before R2, then catalogue order. The rest stay attached as
    `related`, figures intact. Different rows (motor spread vs hover
    throttle, vibration level vs PX4's imbalance detector) stay separate.
    """
    out = [f for f in (finding_for(fs) for fs in scores) if f is not None]
    groups: Dict[str, List[Finding]] = {}
    for i, f in enumerate(out):
        groups.setdefault(f.cause_group, []).append(f)
    collapsed = []
    for members in groups.values():
        ranked = sorted(members, key=lambda f: (f.sort_score, f.tier != "R1", out.index(f)))
        lead = ranked[0]
        lead.related = [{"key": f.key, "figure": f.figure, "value": f.value, "score": f.score, "line": f.line,
                         "kind": f.kind, "comparison": f.comparison} for f in ranked[1:]]
        collapsed.append(lead)
    return sorted(collapsed, key=lambda f: f.sort_score)


# ---------------------------------------------------------------- verdict

def verdict(findings: Sequence[Finding], checks: Optional[List[dict]], n_epoch: int, overall: Optional[float],
            autopilot: str, prompts: Sequence[dict] = ()) -> dict:
    """Spec §3.4 (flight one) or §5.1 (worst three) verdict. Every line is
    built from findings that already passed the §6 checks, and is checked
    again here."""
    if checks is not None:
        counted = [c for c in checks if c["status"] != "unavailable"]
        failed = [c for c in counted if c["status"] not in ("ok", "usage_note")]
        k, m = len(counted) - len(failed), len(counted)
        # "to look at" counts causes (collapsed findings), not failed checks:
        # two GPS checks failing are one thing to look at (spec §5.1).
        j = max(len(findings), 1) if failed else 0
        prefix = "First flight logged: " if n_epoch == 1 else ""
        head = prefix + (f"{k} of {m} checks OK" + (f", {j} to look at" if failed else ""))
        guide = "PX4's" if autopilot == "px4" else "ArduPilot's"
        lines = [findings[0].text if findings else f"Nothing outside {guide} published ranges."]
        lines.append(f"Fly {max(0, 5 - n_epoch)} more to learn your drone's normal.")
    else:
        top = list(findings[:VERDICT_MAX_FINDINGS])
        head = ("In line with your drone's usual" if not top else
                f"{len(findings)} thing{'s' if len(findings) != 1 else ''} to look at")
        lines = [f.text for f in top]
    lines += [p["text"] for p in prompts]
    out = {"headline": head, "lines": lines, "top_findings": [f.key for f in findings[:VERDICT_MAX_FINDINGS]],
           "progress": {"flights": n_epoch, "toward": 5} if n_epoch < 5 else None}
    assert_report_honest(out, "verdict")
    return out


def prompt_text(feature_key: str, direction: str) -> str:
    name = DISPLAY.get(feature_key, Display(feature_key)).name
    return check_claim(f"Did you change something? {name} has been {direction} for 3 flights in a row.")


# Import-time check: every fixed string in this module passes the §6 list.
for _most, _also in CAUSES.values():
    for _t in _most + _also:
        check_claim(_t)
for _d in DISPLAY.values():
    check_claim(_d.name)
for _k in (VIB, MOT, PWR, ATT, POS):
    check_claim(_k)
