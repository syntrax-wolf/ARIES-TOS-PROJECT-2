"""
Persistence for students, submitted runs, and the leaderboard.

SQLite (stdlib, no extra dependency). Table shapes mirror the Postgres/Supabase
design so moving to a hosted database later is a driver swap, not a redesign.

Only server-computed scores are ever stored — see models.py.
"""
import json
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_FILE = BASE_DIR / "playground.db"

# IIT Delhi entry numbers: 4-digit admission year, 2-4 letter department code,
# then a 4-5 digit serial — e.g. 2023CS10123, 2022MT60045.
# Relax this single regex if a cohort uses a different format.
ENTRY_NUMBER_RE = re.compile(r"^\d{4}[A-Z]{2,4}\d{4,5}$")

NAME_MIN_LEN = 2
NAME_MAX_LEN = 60

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    entry_number TEXT NOT NULL UNIQUE,
    created_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id      INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    algorithm       TEXT NOT NULL,
    hyperparams     TEXT NOT NULL DEFAULT '{}',
    summary         TEXT NOT NULL DEFAULT '',
    seed            INTEGER NOT NULL,
    -- server-computed, the only numbers the board ever ranks
    accuracy        REAL NOT NULL,
    macro_f1        REAL NOT NULL,
    train_accuracy  REAL,
    -- what the browser reported, kept for comparison only. Never ranked.
    client_accuracy REAL,
    client_macro_f1 REAL,
    created_at      REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS runs_board_idx
    ON runs (algorithm, macro_f1 DESC, accuracy DESC, created_at ASC);
CREATE INDEX IF NOT EXISTS runs_student_idx
    ON runs (student_id, created_at DESC);
"""


def _connect():
    conn = sqlite3.connect(DB_FILE, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # Every statement is IF NOT EXISTS, so this is a no-op once the schema is
    # there. Running it per connection rather than only at startup means the
    # server recovers on its own if the database file is deleted or replaced
    # underneath it — otherwise every request 500s with "no such table" until
    # someone restarts the process. The parse cost is irrelevant at this scale.
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def _db():
    """One connection per operation — uvicorn runs sync endpoints on a threadpool
    and SQLite connections are not safe to share across threads."""
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _db() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)


# ---------------------------------------------------------------------------
# Students
# ---------------------------------------------------------------------------

def normalize_entry_number(raw: str) -> str:
    return (raw or "").strip().upper().replace(" ", "")


def validate_registration(name: str, entry_number: str) -> tuple[str, str]:
    """Return cleaned (name, entry_number) or raise ValueError with a message
    that is safe to show the student verbatim."""
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
    """Register, or return the existing record for that entry number.

    The entry number is the identity. If it is already known we return the
    originally registered name rather than overwriting it — otherwise anyone
    could rename another student's leaderboard rows by typing their number.
    """
    name, entry_number = validate_registration(name, entry_number)

    with _db() as conn:
        row = conn.execute(
            "SELECT id, name, entry_number FROM students WHERE entry_number = ?",
            (entry_number,),
        ).fetchone()
        if row is not None:
            return {
                "student_id": row["id"],
                "name": row["name"],
                "entry_number": row["entry_number"],
                "returning": True,
            }

        cur = conn.execute(
            "INSERT INTO students (name, entry_number, created_at) VALUES (?, ?, ?)",
            (name, entry_number, time.time()),
        )
        return {
            "student_id": cur.lastrowid,
            "name": name,
            "entry_number": entry_number,
            "returning": False,
        }


def get_student(student_id) -> dict | None:
    if student_id is None:
        return None
    with _db() as conn:
        row = conn.execute(
            "SELECT id, name, entry_number FROM students WHERE id = ?", (student_id,)
        ).fetchone()
    if row is None:
        return None
    return {"student_id": row["id"], "name": row["name"], "entry_number": row["entry_number"]}


def student_stats(student_id: int) -> dict:
    with _db() as conn:
        row = conn.execute(
            """SELECT COUNT(*) AS total_runs,
                      COUNT(DISTINCT algorithm) AS algorithms_tried,
                      MAX(macro_f1) AS best_macro_f1
               FROM runs WHERE student_id = ?""",
            (student_id,),
        ).fetchone()
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
    with _db() as conn:
        cur = conn.execute(
            """INSERT INTO runs (student_id, algorithm, hyperparams, summary, seed,
                                 accuracy, macro_f1, train_accuracy,
                                 client_accuracy, client_macro_f1, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (student_id, algorithm, json.dumps(hyperparams, sort_keys=True), summary, seed,
             accuracy, macro_f1, train_accuracy,
             client_accuracy, client_macro_f1, time.time()),
        )
        return cur.lastrowid


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
    WHERE (:algorithm IS NULL OR r.algorithm = :algorithm)
)
WHERE rn = 1
ORDER BY macro_f1 DESC, accuracy DESC, created_at ASC
"""


def leaderboard(algorithm: str | None = None, limit: int | None = 12) -> list[dict]:
    """Best run per student, ranked. `position` is computed per query rather
    than stored — a stored rank goes stale the moment anyone else submits."""
    with _db() as conn:
        rows = conn.execute(_BOARD_SQL, {"algorithm": algorithm}).fetchall()

    board = [
        {
            "position": i + 1,
            "run_id": r["run_id"],
            "student_id": r["student_id"],
            "name": r["name"],
            "entry_number": r["entry_number"],
            "algorithm": r["algorithm"],
            "hyperparams": json.loads(r["hyperparams"]),
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
    """(position, board_size). Position is None when the run is not that
    student's best — they have already beaten it, so it is not on the board."""
    board = leaderboard(algorithm, limit=None)
    for row in board:
        if row["run_id"] == run_id:
            return row["position"], len(board)
    return None, len(board)


# ---------------------------------------------------------------------------
# Admin views — the full picture behind the public board. Admin-only.
# ---------------------------------------------------------------------------

def admin_overview() -> dict:
    """Headline counts for the dashboard."""
    with _db() as conn:
        totals = conn.execute(
            """SELECT
                 (SELECT COUNT(*) FROM students) AS students,
                 (SELECT COUNT(*) FROM runs)     AS runs,
                 (SELECT COUNT(DISTINCT algorithm) FROM runs) AS algorithms,
                 (SELECT MAX(macro_f1) FROM runs) AS best_macro_f1,
                 (SELECT MAX(created_at) FROM runs) AS last_run_at"""
        ).fetchone()

        per_algo = conn.execute(
            """SELECT algorithm,
                      COUNT(*) AS runs,
                      COUNT(DISTINCT student_id) AS students,
                      MAX(macro_f1) AS best_macro_f1,
                      AVG(macro_f1) AS avg_macro_f1
               FROM runs
               GROUP BY algorithm
               ORDER BY runs DESC"""
        ).fetchall()

    return {
        "students": totals["students"],
        "runs": totals["runs"],
        "algorithms": totals["algorithms"],
        "best_macro_f1": totals["best_macro_f1"],
        "last_run_at": totals["last_run_at"],
        "per_algorithm": [
            {
                "algorithm": r["algorithm"],
                "runs": r["runs"],
                "students": r["students"],
                "best_macro_f1": r["best_macro_f1"],
                "avg_macro_f1": r["avg_macro_f1"],
            }
            for r in per_algo
        ],
    }


def admin_all_runs(algorithm: str | None = None, limit: int = 500) -> list[dict]:
    """Every run, newest first — including a student's non-best runs and the
    client-vs-server score gap, which the public board hides."""
    with _db() as conn:
        rows = conn.execute(
            """SELECT r.id, r.created_at, s.name, s.entry_number, r.algorithm,
                      r.summary, r.hyperparams, r.seed,
                      r.accuracy, r.macro_f1, r.train_accuracy,
                      r.client_accuracy, r.client_macro_f1
               FROM runs r
               JOIN students s ON s.id = r.student_id
               WHERE (:algorithm IS NULL OR r.algorithm = :algorithm)
               ORDER BY r.created_at DESC
               LIMIT :limit""",
            {"algorithm": algorithm, "limit": max(1, min(limit, 5000))},
        ).fetchall()

    return [
        {
            "run_id": r["id"],
            "timestamp": r["created_at"],
            "name": r["name"],
            "entry_number": r["entry_number"],
            "algorithm": r["algorithm"],
            "summary": r["summary"],
            "hyperparams": json.loads(r["hyperparams"]),
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
    """Every registered student with their activity, best-first."""
    with _db() as conn:
        rows = conn.execute(
            """SELECT s.id, s.name, s.entry_number, s.created_at,
                      COUNT(r.id) AS total_runs,
                      COUNT(DISTINCT r.algorithm) AS algorithms_tried,
                      MAX(r.macro_f1) AS best_macro_f1,
                      MAX(r.created_at) AS last_run_at
               FROM students s
               LEFT JOIN runs r ON r.student_id = s.id
               GROUP BY s.id
               ORDER BY best_macro_f1 DESC NULLS LAST, s.created_at ASC""",
        ).fetchall()

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
