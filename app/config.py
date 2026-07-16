"""
Centralized configuration, read from environment variables (or a local .env file).

WHY THIS FILE EXISTS:
The original codebase had the database URL hardcoded directly in database.py
("postgresql://aarushcb@localhost/drone_twin"). That only ever works on one
machine. A mobile app talks to a server over the internet, so the server's
configuration (which database to use, secret keys, etc.) needs to change
per-environment (your laptop vs. the real deployed server) WITHOUT editing code.
This is the standard fix: one settings object, populated from env vars.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Defaults to a local SQLite file so you can run everything with zero setup.
    # In production (Step 2) this becomes a Supabase/Neon Postgres URL instead.
    database_url: str = "sqlite:///./drone_twin.db"

    # Used to sign login tokens (JWTs). MUST be overridden in production —
    # the default here is only safe for local development.
    secret_key: str = "dev-only-secret-change-this-before-deploying"
    algorithm: str = "HS256"

    # 7 days: mobile apps shouldn't force users to log in constantly.
    access_token_expire_minutes: int = 60 * 24 * 7

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
