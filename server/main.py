"""
Sylva leaderboard API.

The browser trains and animates its own models; this server is the referee.
On submit it rebuilds the student's exact train/test split from the seed,
retrains a canonical scikit-learn model, and stores *its own* score. The
client's numbers are recorded alongside for comparison but are never ranked.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import models
import store

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
