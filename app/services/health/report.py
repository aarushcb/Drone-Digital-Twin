"""
The full chain for one drone, oldest flight first: change detection ->
per-feature baseline groups -> admission -> scoring -> findings -> report.

Pure and in-memory. Step 10 will load these inputs from the database (and
persist epochs, admissions and baselines); the logic here doesn't change.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from . import baseline as B
from . import epochs as E
from . import findings as F
from . import norms as N
from . import scoring as S
from .features import FEATURE_CATALOGUE, Feature
from .windowing import FlightWindow

SCHEMA_VERSION = "d1.health.1"


@dataclass
class FlightInput:
    flight_id: int
    autopilot: str                       # px4 | ardupilot
    features: List[Feature]
    window: FlightWindow
    pack_id: Optional[int] = None
    watched: Dict[str, object] = field(default_factory=dict)
    log_meta: Dict[str, object] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    when: Optional[str] = None           # ISO date of the flight (log clock, else upload time); Trends labels


def _observation(fl: FlightInput, f: Feature) -> B.Observation:
    return B.Observation(fl.flight_id, f.value, qualified=fl.window.qualified, valid=f.valid and f.value is not None,
                         absolute_bad=N.is_absolute_bad(f.key, f.value, fl.autopilot, f.detail))


def _enrich_spread(feats: Dict[str, Feature]) -> None:
    """act.spread's finding names the pattern; give it act.pattern's label
    and act.mean's per-motor means so the finding needn't look elsewhere."""
    spread, pattern, mean = feats.get("act.spread"), feats.get("act.pattern"), feats.get("act.mean")
    if spread is None or not spread.valid:
        return
    if pattern is not None and pattern.label:
        spread.detail["pattern_label"] = pattern.label
        for src, dst in (("motor_number", "pattern_motor_number"), ("high_motor_numbers", "pattern_high_motors"),
                         ("toward", "pattern_toward")):
            if src in pattern.detail:
                spread.detail[dst] = pattern.detail[src]
    if mean is not None and mean.valid:
        spread.detail["per_motor"] = mean.detail.get("per_motor")


def detect_events(flights: Sequence[FlightInput], declared: Sequence[E.ConfigEvent]) -> List[E.ConfigEvent]:
    """Declared events plus watched-parameter changes between consecutive
    flights, in flight order."""
    events = list(declared)
    for i in range(1, len(flights)):
        events += E.events_from_params(i, flights[i - 1].watched, flights[i].watched)
    return sorted(events, key=lambda e: e.flight_index)


def analyse(flights: Sequence[FlightInput], declared: Sequence[E.ConfigEvent] = ()) -> List[dict]:
    """One d1.health.1 report per flight."""
    events = detect_events(flights, declared)
    feats_by_flight = []
    for fl in flights:
        fmap = {f.key: f for f in fl.features}
        _enrich_spread(fmap)
        feats_by_flight.append(fmap)

    def epoch_key(idx: Optional[int]) -> Optional[str]:
        return None if idx is None else events[idx].key(flights[events[idx].flight_index].flight_id)

    groups = [{k: B.baseline_group(k, epoch_key(E.feature_epoch(k, i, events)), fl.pack_id)
               for k, *_ in FEATURE_CATALOGUE} for i, fl in enumerate(flights)]
    return [_report(i, flights, feats_by_flight, groups, events) for i in range(len(flights))]


def _report(i, flights, feats_by_flight, groups, events) -> dict:
    fl = flights[i]
    scores: List[S.FeatureScore] = []
    prompts: List[dict] = []
    for key, *_ in FEATURE_CATALOGUE:
        f = feats_by_flight[i][key]
        g = groups[i][key]
        history = [_observation(flights[j], feats_by_flight[j][key]) for j in range(i + 1) if groups[j][key] == g]
        prior_res = B.process_history(history[:-1], key)
        cur = history[-1]
        cur.admitted, cur.quarantine_reason = B.admission(cur, prior_res.admitted, key)
        scores.append(S.score_feature(f, fl.autopilot, g, prior_res.observations, cur))
        if key in B.DIRECTION:
            p = E.inferred_change(prior_res.observations + [cur], key)
            if p is not None:
                prompts.append({"feature": key, "direction": p.direction, "flight_ids": p.flight_ids,
                                "values": p.values, "baseline_median": p.baseline_median,
                                "max_pairwise_z": p.max_pairwise_z, "text": F.prompt_text(key, p.direction)})

    roll = S.rollup(scores)
    by_key = {fs.key: fs for fs in scores}
    epoch_idx = E.latest_epoch(i, events)
    epoch_start = 0 if epoch_idx is None else events[epoch_idx].flight_index
    epoch_key = None if epoch_idx is None else events[epoch_idx].key(flights[epoch_start].flight_id)
    n_epoch = sum(1 for j in range(epoch_start, i + 1) if flights[j].window.qualified)
    cold_start = roll.overall is None and n_epoch <= 2
    checks = S.build_checks(by_key, fl.autopilot) if cold_start else None
    findings = F.generate(scores)
    new_events = [e for e in events if e.flight_index == i]
    w = fl.window

    warnings = list(fl.warnings) + list(w.warnings)
    if not w.steady_qualified and w.qualified:
        warnings.append("insufficient_steady_flight")
    if new_events:
        warnings.append("new_epoch")

    report = {
        "schema_version": SCHEMA_VERSION,
        "run": {"flight_id": fl.flight_id, "flight_number": i + 1, "pack_id": fl.pack_id,
                "epoch_id": epoch_key, "autopilot": fl.autopilot},
        "real_log_meta": fl.log_meta,
        "flight_window": {"qualified": w.qualified, "reasons": w.reasons,
                          "airborne_start_s": None if w.airborne_start_us is None else round(w.airborne_start_us / 1e6, 2),
                          "airborne_end_s": None if w.airborne_end_us is None else round(w.airborne_end_us / 1e6, 2),
                          "airborne_s": w.airborne_s, "airborne_source": w.airborne_source, "steady_s": w.steady_s,
                          "phases": w.phase_seconds},
        "warnings": warnings,
        "spectral_source": (by_key["vib.band_struct"].feature.detail or {}).get("source"),
        "coverage": roll.coverage,
        "overall_score": roll.overall,
        "score_band": roll.score_band,
        "subsystem_scores": roll.subsystem_scores,
        "subsystems_scored": roll.scored_subsystems,
        "overall_cap": roll.overall_cap,
        "baseline": {"epoch_id": epoch_key, "epoch_started_flight": epoch_start + 1, "qualifying_flights": n_epoch,
                     "pack_id": fl.pack_id,
                     "mode_summary": {m: sum(1 for fs in scores if fs.status == "scored" and fs.mode == m)
                                      for m in ("absolute", "blended", "personal")}},
        "param_diff": [c for e in new_events if e.source == "params" for c in e.detail.get("changes", [])],
        "epoch_events": [{"flight_number": e.flight_index + 1, "epoch_id": e.key(fl.flight_id),
                          "change_type": e.change_type, "source": e.source,
                          "note": e.note, "features_reset": sorted(e.reset_features)} for e in new_events],
        "features": [_feature_json(fs) for fs in scores],
        "findings": [f.to_json() for f in findings],
        "verdict": F.verdict(findings, checks, n_epoch, roll.overall, fl.autopilot, prompts),
        "checks": checks,
        "change_prompts": prompts,
        "logging_gaps": S.logging_gaps([fs.feature for fs in scores]),
    }
    F.assert_report_honest(report)
    return report


def _r(x, d=4):
    return None if x is None else round(float(x), d)


def _baseline_json(b: Optional[B.Baseline], group: str) -> Optional[dict]:
    if b is None:
        return None
    return {"kind": b.kind, "median": _r(b.median), "mad": _r(b.mad), "spread": _r(b.spread), "n": b.n,
            "flight_ids": b.flight_ids, "group": group}


def _feature_json(fs: S.FeatureScore) -> dict:
    f = fs.feature
    return {
        "key": f.key, "subsystem": f.subsystem, "value": _r(f.value, 6), "label": f.label, "unit": f.unit,
        # Display copy from the same table the findings use, so the app never re-formats a number.
        "name": F.DISPLAY[f.key].name if f.key in F.DISPLAY else None,
        "figure": F.figure(f.value, F.DISPLAY[f.key]) if f.key in F.DISPLAY and f.value is not None else None,
        "tier": f.tier, "valid": f.valid, "reason": f.reason, "status": fs.status, "group": fs.group,
        "unscored_reason": fs.unscored_reason, "mode": fs.mode, "basis": fs.basis, "n": fs.n,
        "baseline": _baseline_json(fs.rolling, fs.group), "reference": _baseline_json(fs.reference, fs.group),
        "z": _r(fs.z, 3), "score_baseline": _r(fs.s_baseline, 1),
        "score_absolute": _r(fs.absolute.score, 1), "absolute_status": fs.absolute.status,
        "weight": _r(fs.weight, 3), "score": fs.score, "capped": fs.capped, "confidence": fs.confidence,
        "admitted": fs.admitted, "quarantine_reason": fs.quarantine_reason, "detail": _jsonable(f.detail),
    }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item"):           # numpy scalar
        return obj.item()
    return obj
