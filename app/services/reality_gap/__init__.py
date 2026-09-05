"""
D1 -- Reality-Gap Quantifier.

Spec: ~/Desktop/drone_twin_app/assets/docs/specs/D1-reality-gap-quantifier-spec.md

Build status: PASS 1 of N. Only the ingestion half of the pipeline exists
so far -- format detection + parsing to the canonical schema (parsers.py,
spec 5.2) and alignment/resampling/segment selection (align.py, spec
5.3-5.5). Metrics (spec 3), scoring (spec 4), SysID hints (spec 6),
the API endpoints (spec 8), the database models and their Alembic
migration, and all Flutter work are deliberately NOT built yet and are
not stubbed -- there are no placeholder modules for them, so an import
error is the honest signal that a later pass hasn't happened rather than
a silently-empty result.
"""
