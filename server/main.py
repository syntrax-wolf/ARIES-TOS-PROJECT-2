"""
Sylva leaderboard API.

The browser trains and animates its own models; this server is the referee.
On submit it rebuilds the student's exact train/test split from the seed,
retrains a canonical scikit-learn model, and stores *its own* score. The
client's numbers are recorded alongside for comparison but are never ranked.
"""
import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from server/.env into the environment, if present.
    Keeps the admin secrets out of the shell and out of git (.env is ignored).
    Real environment variables always win, so this never clobbers an explicit
    export."""
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()  # must run before auth reads its config

import auth      # noqa: E402  (import after .env is loaded)
import models    # noqa: E402
import store     # noqa: E402

app = FastAPI(title="Sylva Leaderboard API")
store.init_db()

# The Vite dev server runs on a different origin during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Students
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    name: str
    entry_number: str


@app.post("/api/register")
def register(req: RegisterRequest):
    try:
        student = store.register_student(req.name, req.entry_number)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    student["stats"] = store.student_stats(student["student_id"])
    return student


@app.get("/api/student/{student_id}")
def get_student(student_id: int):
    student = store.get_student(student_id)
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found.")
    student["stats"] = store.student_stats(student_id)
    return student


# ---------------------------------------------------------------------------
# Submissions
# ---------------------------------------------------------------------------

class SubmitRequest(BaseModel):
    student_id: int
    algorithm: str
    seed: int
    hyperparams: dict = Field(default_factory=dict)
    # What the browser's own model scored. Stored for comparison, never ranked.
    client_accuracy: float | None = None
    client_macro_f1: float | None = None


@app.post("/api/submit")
def submit(req: SubmitRequest):
    student = store.get_student(req.student_id)
    if student is None:
        raise HTTPException(
            status_code=400,
            detail="Unknown student. Register with your name and entry number first.",
        )

    if req.algorithm not in models.RATED_ALGORITHMS:
        raise HTTPException(
            status_code=400,
            detail=f"'{req.algorithm}' is not a rated algorithm.",
        )

    hyperparams = models.clamp_hyperparams(req.hyperparams)

    try:
        score = models.evaluate(req.algorithm, hyperparams, req.seed)
    except Exception as exc:  # noqa: BLE001 — surface the cause, don't 500 silently
        raise HTTPException(status_code=400, detail=f"Could not evaluate run: {exc}")

    summary = models.summarize(req.algorithm, hyperparams)
    run_id = store.record_run(
        student_id=student["student_id"],
        algorithm=req.algorithm,
        hyperparams=hyperparams,
        summary=summary,
        seed=req.seed,
        accuracy=score.accuracy,
        macro_f1=score.macro_f1,
        train_accuracy=score.train_accuracy,
        client_accuracy=req.client_accuracy,
        client_macro_f1=req.client_macro_f1,
    )
    position, total = store.rank_of_run(run_id)

    return {
        "run_id": run_id,
        "student": student,
        "algorithm": req.algorithm,
        "algorithm_label": models.ALGORITHM_LABELS[req.algorithm],
        "hyperparams": hyperparams,
        "summary": summary,
        "seed": req.seed,
        "official": {
            "accuracy": score.accuracy,
            "macro_f1": score.macro_f1,
            "train_accuracy": score.train_accuracy,
            "n_train": score.n_train,
            "n_test": score.n_test,
        },
        "client": {
            "accuracy": req.client_accuracy,
            "macro_f1": req.client_macro_f1,
        },
        "position": position,
        "total": total,
    }


@app.get("/api/leaderboard")
def leaderboard(algorithm: str | None = None, limit: int = 12):
    return store.leaderboard(algorithm, limit=max(1, min(limit, 100)))


@app.get("/api/health")
def health():
    return {"ok": True}


# ---------------------------------------------------------------------------
# Admin dashboard — gated by the admin's Kerberos id + a separate password.
#
# Not part of the student API. The page and its data endpoints live under
# /admin, are never bundled into the student frontend, and every data route
# requires a valid signed session cookie.
# ---------------------------------------------------------------------------

def require_admin(request: Request):
    token = request.cookies.get(auth.SESSION_COOKIE)
    if not auth.valid_session(token):
        raise HTTPException(status_code=401, detail="Admin sign-in required.")


class AdminLoginRequest(BaseModel):
    kerberos: str
    password: str


@app.post("/admin/api/login")
def admin_login(req: AdminLoginRequest, response: Response):
    try:
        token = auth.login(req.kerberos, req.password)
    except auth.LoginError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    response.set_cookie(
        key=auth.SESSION_COOKIE,
        value=token,
        max_age=auth.SESSION_TTL_SECONDS,
        httponly=True,      # not readable from JS
        samesite="strict",  # not sent on cross-site requests
        # secure=True omitted so it works over http://localhost; enable behind HTTPS.
    )
    return {"ok": True}


@app.post("/admin/api/logout")
def admin_logout(response: Response):
    response.delete_cookie(auth.SESSION_COOKIE)
    return {"ok": True}


@app.get("/admin/api/session")
def admin_session(request: Request):
    token = request.cookies.get(auth.SESSION_COOKIE)
    return {"authenticated": auth.valid_session(token), "configured": auth.is_configured()}


@app.get("/admin/api/overview", dependencies=[Depends(require_admin)])
def admin_overview():
    return store.admin_overview()


@app.get("/admin/api/runs", dependencies=[Depends(require_admin)])
def admin_runs(algorithm: str | None = None, limit: int = 500):
    return store.admin_all_runs(algorithm, limit=limit)


@app.get("/admin/api/students", dependencies=[Depends(require_admin)])
def admin_students():
    return store.admin_students()


@app.get("/admin")
def admin_page():
    return FileResponse(BASE_DIR / "admin.html")
