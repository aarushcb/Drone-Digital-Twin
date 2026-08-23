import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# So `from app...` imports work when Alembic is invoked from the project
# root (the normal case: `alembic revision`/`alembic upgrade` run from
# ~/Desktop/Drone-Digital-Twin) or from anywhere else.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# WHY THIS IMPORTS THE APP'S OWN SETTINGS INSTEAD OF READING
# alembic.ini's sqlalchemy.url: one source of truth for "which database
# are we talking to" -- app/config.py's Settings already reads
# DATABASE_URL from the environment/.env (local SQLite by default,
# Neon Postgres in production, same as the running API server uses).
# Duplicating that URL into alembic.ini would just be a second place for
# it to drift out of sync.
from app.config import settings

# Import Base AND every model module so they register on Base.metadata --
# SQLAlchemy only knows about a model once its module has been imported
# somewhere; without this, autogenerate would see an empty target schema
# and try to DROP every real table. Mirrors main.py's own import list.
from app.database.database import Base
from app.models import user, drone, telemetry, scene_object  # noqa: F401 -- import so tables register

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config
config.set_main_option("sqlalchemy.url", settings.database_url)

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Excludes tables that exist in the live database but aren't part of
# this app's own schema, from autogenerate's diffing -- specifically
# Neon's own auto-provisioned "playing_with_neon" demo table, which
# every fresh Neon project gets automatically and isn't something this
# app created or manages. Without this, every future `alembic revision
# --autogenerate` run against the real Neon database would spuriously
# propose dropping it (confirmed directly while setting up the baseline
# migration -- see the baseline revision's commit message).
_IGNORED_TABLES = {"playing_with_neon"}


def _include_name(name, type_, parent_names):
    if type_ == "table" and name in _IGNORED_TABLES:
        return False
    return True


# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_name=_include_name,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_name=_include_name,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
