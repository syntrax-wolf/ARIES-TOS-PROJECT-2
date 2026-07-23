"""
Storage backend selector.

If SUPABASE_DB_URL (or DATABASE_URL) is set, use Postgres/Supabase; otherwise
fall back to local SQLite so development works with no credentials. Both
modules expose the identical public API, so main.py imports `store` either way.
"""
import os

if os.environ.get("SUPABASE_DB_URL") or os.environ.get("DATABASE_URL"):
    from store_pg import *  # noqa: F401,F403
    BACKEND = "postgres"
else:
    from store_sqlite import *  # noqa: F401,F403
    BACKEND = "sqlite"
