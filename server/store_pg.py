"""
Postgres/Supabase backend — the same public API as store_sqlite.py.

Follows the UAI pattern: one Supabase Postgres, this project's tables kept in
their own schema ("sylva") so they never collide with anything else in the
project, connection string read from SUPABASE_DB_URL in the environment/.env.

Timestamps are stored as epoch seconds (double precision) rather than
TIMESTAMPTZ, so every API response is byte-identical to the SQLite backend and
the frontend needs no changes.

Selected as the active backend by store.py whenever SUPABASE_DB_URL is set.
"""
import json
import os
import re
import time

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool

SCHEMA = "sylva"

ENTRY_NUMBER_RE = re.compile(r"^\d{4}[A-Z]{2,4}\d{4,5}$")
NAME_MIN_LEN = 2
NAME_MAX_LEN = 60

DDL = f"""
CREATE SCHEMA IF NOT EXISTS {SCHEMA};

CREATE TABLE IF NOT EXISTS {SCHEMA}.students (
    id            BIGGENERATED_PLACEHOLDER,
    name          TEXT NOT NULL,
    entry_number  TEXT NOT NULL UNIQUE,
    created_at    DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.runs (
    id              BIGGENERATED_PLACEHOLDER,
    student_id      BIGINT NOT NULL REFERENCES {SCHEMA}.students(id) ON DELETE CASCADE,
    algorithm       TEXT NOT NULL,
    hyperparams     JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    summary         TEXT NOT NULL DEFAULT '',
    seed            BIGINT NOT NULL,
    accuracy        DOUBLE PRECISION NOT NULL,
    macro_f1        DOUBLE PRECISION NOT NULL,
    train_accuracy  DOUBLE PRECISION,
    client_accuracy DOUBLE PRECISION,
    client_macro_f1 DOUBLE PRECISION,
    created_at      DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS runs_board_idx
    ON {SCHEMA}.runs (algorithm, macro_f1 DESC, accuracy DESC, created_at ASC);
CREATE INDEX IF NOT EXISTS runs_student_idx
    ON {SCHEMA}.runs (student_id, created_at DESC);
""".replace("BIGGENERATED_PLACEHOLDER", "BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY")


_pool: ThreadedConnectionPool | None = None


def _dsn() -> str:
    url = os.environ.get("SUPABASE_DB_URL") or os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("SUPABASE_DB_URL is not set.")
    return url


def _get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        # search_path scopes every query to our schema; public kept for extensions.
        _pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=8,
            dsn=_dsn(),
            options=f"-c search_path={SCHEMA},public",
            connect_timeout=15,
        )
    return _pool


class _Conn:
    """Borrow a pooled connection, commit on success, always return it."""

    def __enter__(self):
        self.conn = _get_pool().getconn()
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        _get_pool().putconn(self.conn)
        return False


def _cursor(conn):
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)


def init_db():
    with _Conn() as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)


# ---------------------------------------------------------------------------
# Students
# ---------------------------------------------------------------------------

def normalize_entry_number(raw: str) -> str:
    return (raw or "").strip().upper().replace(" ", "")


def validate_registration(name: str, entry_number: str) -> tuple[str, str]:
    name = (name or "").strip()
    entry_number = normalize_entry_number(entry_number)
    if len(name) < NAME_MIN_LEN:
        raise ValueError("Please enter your full name.")
    if len(name) > NAME_MAX_LEN:
        raise ValueError(f"Name must be at most {NAME_MAX_LEN} characters.")
    if not entry_number:
        raise ValueError("Please enter your entry number.")
    if not ENTRY_NUMBER_RE.match(entry_number):
        raise ValueError("That doesn't look like an entry number. Expected format: 2023CS10123.")
    return name, entry_number


def register_student(name: str, entry_number: str) -> dict:
    name, entry_number = validate_registration(name, entry_number)
    with _Conn() as conn, _cursor(conn) as cur:
        cur.execute(
            "SELECT id, name, entry_number FROM students WHERE entry_number = %s",
            (entry_number,),
        )
        row = cur.fetchone()
        if row is not None:
            return {
                "student_id": row["id"],
                "name": row["name"],
                "entry_number": row["entry_number"],
                "returning": True,
            }
        cur.execute(
            "INSERT INTO students (name, entry_number, created_at) VALUES (%s, %s, %s) RETURNING id",
            (name, entry_number, time.time()),
        )
        new_id = cur.fetchone()["id"]
        return {"student_id": new_id, "name": name, "entry_number": entry_number, "returning": False}


def get_student(student_id) -> dict | None:
    if student_id is None:
        return None
    with _Conn() as conn, _cursor(conn) as cur:
        cur.execute("SELECT id, name, entry_number FROM students WHERE id = %s", (student_id,))
        row = cur.fetchone()
    if row is None:
        return None
    return {"student_id": row["id"], "name": row["name"], "entry_number": row["entry_number"]}


def student_stats(student_id: int) -> dict:
    with _Conn() as conn, _cursor(conn) as cur:
        cur.execute(
            """SELECT COUNT(*) AS total_runs,
                      COUNT(DISTINCT algorithm) AS algorithms_tried,
                      MAX(macro_f1) AS best_macro_f1
               FROM runs WHERE student_id = %s""",
            (student_id,),
        )
        row = cur.fetchone()
    return {
        "total_runs": row["total_runs"],
        "algorithms_tried": row["algorithms_tried"],
        "best_macro_f1": row["best_macro_f1"],
    }


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def record_run(student_id: int, algorithm: str, hyperparams: dict, summary: str,
               seed: int, accuracy: float, macro_f1: float, train_accuracy: float,
               client_accuracy: float | None, client_macro_f1: float | None) -> int:
    with _Conn() as conn, _cursor(conn) as cur:
        cur.execute(
            """INSERT INTO runs (student_id, algorithm, hyperparams, summary, seed,
                                 accuracy, macro_f1, train_accuracy,
                                 client_accuracy, client_macro_f1, created_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (student_id, algorithm, json.dumps(hyperparams, sort_keys=True), summary, seed,
             accuracy, macro_f1, train_accuracy, client_accuracy, client_macro_f1, time.time()),
        )
        return cur.fetchone()["id"]


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

_BOARD_SQL = """
SELECT run_id, student_id, name, entry_number, algorithm, hyperparams, summary,
       seed, accuracy, macro_f1, train_accuracy, created_at
FROM (
    SELECT r.id AS run_id, r.student_id, s.name, s.entry_number, r.algorithm,
           r.hyperparams, r.summary, r.seed, r.accuracy, r.macro_f1,
           r.train_accuracy, r.created_at,
           ROW_NUMBER() OVER (
               PARTITION BY r.student_id
               ORDER BY r.macro_f1 DESC, r.accuracy DESC, r.created_at ASC
           ) AS rn
    FROM runs r
    JOIN students s ON s.id = r.student_id
    WHERE (%(algorithm)s IS NULL OR r.algorithm = %(algorithm)s)
) ranked
WHERE rn = 1
ORDER BY macro_f1 DESC, accuracy DESC, created_at ASC
"""


def leaderboard(algorithm: str | None = None, limit: int | None = 12) -> list[dict]:
    with _Conn() as conn, _cursor(conn) as cur:
        cur.execute(_BOARD_SQL, {"algorithm": algorithm})
        rows = cur.fetchall()
    board = [
        {
            "position": i + 1,
            "run_id": r["run_id"],
            "student_id": r["student_id"],
            "name": r["name"],
            "entry_number": r["entry_number"],
            "algorithm": r["algorithm"],
            "hyperparams": r["hyperparams"],
            "summary": r["summary"],
            "seed": r["seed"],
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
            "train_accuracy": r["train_accuracy"],
            "timestamp": r["created_at"],
        }
        for i, r in enumerate(rows)
    ]
    return board if limit is None else board[:limit]


def rank_of_run(run_id: int, algorithm: str | None = None) -> tuple[int | None, int]:
    board = leaderboard(algorithm, limit=None)
    for row in board:
        if row["run_id"] == run_id:
            return row["position"], len(board)
    return None, len(board)


# ---------------------------------------------------------------------------
# Admin views
# ---------------------------------------------------------------------------

def admin_overview() -> dict:
    with _Conn() as conn, _cursor(conn) as cur:
        cur.execute(
            """SELECT
                 (SELECT COUNT(*) FROM students) AS students,
                 (SELECT COUNT(*) FROM runs)     AS runs,
                 (SELECT COUNT(DISTINCT algorithm) FROM runs) AS algorithms,
                 (SELECT MAX(macro_f1) FROM runs) AS best_macro_f1,
                 (SELECT MAX(created_at) FROM runs) AS last_run_at"""
        )
        totals = cur.fetchone()
        cur.execute(
            """SELECT algorithm,
                      COUNT(*) AS runs,
                      COUNT(DISTINCT student_id) AS students,
                      MAX(macro_f1) AS best_macro_f1,
                      AVG(macro_f1) AS avg_macro_f1
               FROM runs GROUP BY algorithm ORDER BY runs DESC"""
        )
        per_algo = cur.fetchall()
    return {
        "students": totals["students"],
        "runs": totals["runs"],
        "algorithms": totals["algorithms"],
        "best_macro_f1": totals["best_macro_f1"],
        "last_run_at": totals["last_run_at"],
        "per_algorithm": [dict(r) for r in per_algo],
    }


def admin_all_runs(algorithm: str | None = None, limit: int = 500) -> list[dict]:
    with _Conn() as conn, _cursor(conn) as cur:
        cur.execute(
            """SELECT r.id, r.created_at, s.name, s.entry_number, r.algorithm,
                      r.summary, r.hyperparams, r.seed,
                      r.accuracy, r.macro_f1, r.train_accuracy,
                      r.client_accuracy, r.client_macro_f1
               FROM runs r
               JOIN students s ON s.id = r.student_id
               WHERE (%(algorithm)s IS NULL OR r.algorithm = %(algorithm)s)
               ORDER BY r.created_at DESC
               LIMIT %(limit)s""",
            {"algorithm": algorithm, "limit": max(1, min(limit, 5000))},
        )
        rows = cur.fetchall()
    return [
        {
            "run_id": r["id"],
            "timestamp": r["created_at"],
            "name": r["name"],
            "entry_number": r["entry_number"],
            "algorithm": r["algorithm"],
            "summary": r["summary"],
            "hyperparams": r["hyperparams"],
            "seed": r["seed"],
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
            "train_accuracy": r["train_accuracy"],
            "client_accuracy": r["client_accuracy"],
            "client_macro_f1": r["client_macro_f1"],
        }
        for r in rows
    ]


def admin_students() -> list[dict]:
    with _Conn() as conn, _cursor(conn) as cur:
        cur.execute(
            """SELECT s.id, s.name, s.entry_number, s.created_at,
                      COUNT(r.id) AS total_runs,
                      COUNT(DISTINCT r.algorithm) AS algorithms_tried,
                      MAX(r.macro_f1) AS best_macro_f1,
                      MAX(r.created_at) AS last_run_at
               FROM students s
               LEFT JOIN runs r ON r.student_id = s.id
               GROUP BY s.id
               ORDER BY MAX(r.macro_f1) DESC NULLS LAST, s.created_at ASC"""
        )
        rows = cur.fetchall()
    return [
        {
            "student_id": r["id"],
            "name": r["name"],
            "entry_number": r["entry_number"],
            "registered_at": r["created_at"],
            "total_runs": r["total_runs"],
            "algorithms_tried": r["algorithms_tried"],
            "best_macro_f1": r["best_macro_f1"],
            "last_run_at": r["last_run_at"],
        }
        for r in rows
    ]
