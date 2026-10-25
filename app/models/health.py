"""
D1 Health tables -- health spec §7 and §9 step 6.

Added via Alembic (CLAUDE.md rule 2), never by create_all on Neon.

DELIBERATELY NOT IMPORTED BY main.py: main.py runs Base.metadata.create_all()
on startup, which creates any table whose model has been imported. If these
models were imported on the startup path before the migration has been
applied to Neon, the next deploy would create the tables itself, bypassing
Alembic, and `alembic upgrade head` would then fail on "table exists".
Only alembic/env.py imports this module until the migration is applied.

Deleting a drone cascades through every table here at the database level
(ondelete="CASCADE"); no ORM relationship on Drone is needed.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint,
)

from app.database.database import Base


def _now():
    return datetime.now(timezone.utc)


class Pack(Base):
    """A battery pack. Power features keep one baseline per pack (spec §2.4)."""
    __tablename__ = "packs"

    id = Column(Integer, primary_key=True, index=True)
    drone_id = Column(Integer, ForeignKey("drones.id", ondelete="CASCADE"), nullable=False, index=True)
    label = Column(String, nullable=False)                 # what the student calls it, e.g. "Pack B"
    cell_count = Column(Integer, nullable=True)
    capacity_mah = Column(Float, nullable=True)
    serial_number = Column(String, nullable=True)          # smart packs only (battery_status.serial_number)
    first_used_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=_now, nullable=False)


class ConfigEpoch(Base):
    """A configuration epoch: a span of flights with no deliberate change
    (spec §2.4). A new epoch resets the baseline of the features its change
    type affects."""
    __tablename__ = "config_epochs"

    id = Column(Integer, primary_key=True, index=True)
    drone_id = Column(Integer, ForeignKey("drones.id", ondelete="CASCADE"), nullable=False, index=True)
    started_at = Column(DateTime, nullable=False)
    change_type = Column(String, nullable=False)           # initial | props | motors_escs | pack | frame | payload | firmware_tuning | gps
    source = Column(String, nullable=False)                # initial | user | params | inferred
    note = Column(Text, nullable=True)                     # free text, incl. optional prop pitch/blade count
    affected_features = Column(JSON, nullable=True)        # feature keys whose baseline this epoch reset
    params_snapshot = Column(JSON, nullable=True)          # watched parameters at the epoch's first flight
    created_at = Column(DateTime, default=_now, nullable=False)


class Flight(Base):
    """One uploaded flight log and its windowing result."""
    __tablename__ = "flights"
    __table_args__ = (UniqueConstraint("drone_id", "log_sha256", name="uq_flights_drone_sha256"),)

    id = Column(Integer, primary_key=True, index=True)
    drone_id = Column(Integer, ForeignKey("drones.id", ondelete="CASCADE"), nullable=False, index=True)
    epoch_id = Column(Integer, ForeignKey("config_epochs.id", ondelete="CASCADE"), nullable=False, index=True)
    pack_id = Column(Integer, ForeignKey("packs.id", ondelete="SET NULL"), nullable=True, index=True)

    flown_at = Column(DateTime, nullable=True)             # log start time (UTC) when the log carries it
    uploaded_at = Column(DateTime, default=_now, nullable=False)
    log_filename = Column(String, nullable=False)
    log_sha256 = Column(String(64), nullable=False)        # spec 9 case 14: same file twice
    autopilot = Column(String, nullable=False)             # px4 | ardupilot; baselines never mix them (spec §6.12)
    firmware_version = Column(String, nullable=True)
    airframe = Column(String, nullable=True)

    qualified = Column(Boolean, nullable=False)
    qualify_reasons = Column(JSON, nullable=True)
    airborne_s = Column(Float, nullable=True)
    steady_s = Column(Float, nullable=True)
    airborne_source = Column(String, nullable=True)        # land_detector | altitude_threshold | none
    phase_seconds = Column(JSON, nullable=True)
    topic_info = Column(JSON, nullable=True)               # per-topic presence and logging rate
    motor_info = Column(JSON, nullable=True)               # motor count, mapping source
    watched_params = Column(JSON, nullable=True)           # spec §2.4 epoch-detection parameters
    warnings = Column(JSON, nullable=True)
    takeoff_mass_kg = Column(Float, nullable=True)         # optional, unscored context (spec §7)


class FlightFeature(Base):
    """One feature value for one flight -- the source of truth baselines
    are computed from. Quarantined values stay here, never deleted."""
    __tablename__ = "flight_features"
    __table_args__ = (UniqueConstraint("flight_id", "feature_key", name="uq_flight_features_flight_key"),)

    id = Column(Integer, primary_key=True, index=True)
    flight_id = Column(Integer, ForeignKey("flights.id", ondelete="CASCADE"), nullable=False, index=True)
    feature_key = Column(String, nullable=False, index=True)
    subsystem = Column(String, nullable=False)
    tier = Column(String(2), nullable=False)               # R1 | R2 | R3
    unit = Column(String, nullable=True)
    value = Column(Float, nullable=True)
    label = Column(String, nullable=True)                  # categorical features (act.pattern)
    reason = Column(String, nullable=True)                 # why value/label is null
    detail = Column(JSON, nullable=True)
    baseline_group = Column(String, nullable=True)         # epoch:<id> | pack:<id>
    admitted = Column(Boolean, nullable=True)              # null until the baseline engine has run
    quarantine_reason = Column(String, nullable=True)


class FeatureBaseline(Base):
    """Cache of the latest rolling and frozen-reference baseline per group
    and feature (spec §9 step 6). Rebuildable from flight_features."""
    __tablename__ = "feature_baselines"
    __table_args__ = (
        UniqueConstraint("drone_id", "baseline_group", "feature_key", "kind", name="uq_feature_baselines_group"),
    )

    id = Column(Integer, primary_key=True, index=True)
    drone_id = Column(Integer, ForeignKey("drones.id", ondelete="CASCADE"), nullable=False, index=True)
    baseline_group = Column(String, nullable=False)        # epoch:<id> | pack:<id>
    feature_key = Column(String, nullable=False)
    kind = Column(String, nullable=False)                  # rolling | reference
    median = Column(Float, nullable=False)
    mad = Column(Float, nullable=False)
    spread = Column(Float, nullable=False)                 # max(1.4826 * MAD, floor)
    n = Column(Integer, nullable=False)
    flight_ids = Column(JSON, nullable=False)
    computed_at = Column(DateTime, default=_now, nullable=False)
