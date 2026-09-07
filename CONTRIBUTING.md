# Contributing — Adding a New Algorithm

Thanks for wanting to extend the ARIES ML Playground! This guide is a **step-by-step template** for
adding a new classification algorithm (e.g. Random Forest, SVM, Logistic Regression) so it shows up
in the UI, trains live, and lands on the leaderboard — using the same patterns the existing Decision
Tree and KNN trainers already follow.

> **Golden rule:** a new algorithm must return the same **snapshot** shape every existing algorithm
> returns, so the frontend can animate it without special-casing. If you produce valid `boundary`
> grids per step, most of the UI works for free.

---

## The contract at a glance

A trainer is a function that returns `(snapshots, final, advice, params_summary)`:

- **`snapshots`** — a list, one per animation step. Each **must** contain:
  - `step_label` (str) — shown in the UI, e.g. `"Depth 3"` or `"k = 5"`.
  - `train_accuracy`, `test_accuracy` (float, 0–1).
  - `boundary` — the decision-surface grid from `_decision_boundary_grid(model, x_range, y_range)`.
  - …plus any algorithm-specific extras (the tree passes `tree`, KNN passes `k`/`weights`).
- **`final`** — usually `snapshots[-1]`.
- **`advice`** — a list of `{ "type": "success" | "warning" | "info", "text": "..." }`.
- **`params_summary`** — a short string for the leaderboard, e.g. `"depth=3"` or `"k=5, uniform"`.

---

## Step 1 — Add request fields (if needed)

Only if your algorithm needs new hyperparameters. In `backend/main.py`, extend `TrainRequest`:

```python
class TrainRequest(BaseModel):
    dataset_id: str
    algorithm: str = "decision_tree"
    # ... existing fields ...
    # --- your new algorithm's hyperparameters, with sensible defaults ---
    n_estimators: int = 50          # example: Random Forest
```

Keep defaults valid so the endpoint never crashes when a field is omitted.

## Step 2 — Write the trainer function

Add a function next to `_train_decision_tree` / `_train_knn`. Copy this skeleton:

```python
def _train_random_forest(req, dataset, x_range, y_range):
    X_train, X_test = dataset["X_train"], dataset["X_test"]
    y_train, y_test = dataset["y_train"], dataset["y_test"]

    # Clamp user input to a safe range — never trust the raw value.
    n_max = max(1, min(req.n_estimators, 100))
    snapshots = []

    # Produce ONE snapshot per animation step. Grow the model so the boundary
    # visibly "improves" — e.g. more estimators, deeper, larger k, more epochs.
    for n in range(1, n_max + 1):
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)

        snapshots.append({
            "step_label": f"{n} trees",
            "n_estimators": n,
            "train_accuracy": round(model.score(X_train, y_train), 4),
            "test_accuracy": round(model.score(X_test, y_test), 4),
            "boundary": _decision_boundary_grid(model, x_range, y_range),
        })

    final = snapshots[-1]
    advice = _generate_advice_random_forest(final["train_accuracy"],
                                            final["test_accuracy"], n_max)
    params_summary = f"trees={n_max}"
    return snapshots, final, advice, params_summary
```

**Guidelines**
- Always **clamp** hyperparameters (`max(1, min(value, CAP))`) so a malicious/huge value can't hang
  the server. `_decision_boundary_grid` runs `model.predict` over a `60×60` grid per snapshot, so
  keep the number of snapshots reasonable (≤ ~25).
- Use `random_state=42` where the estimator supports it, for reproducible leaderboard scores.
- If your model doesn't have a natural "step" (e.g. a one-shot fit), you can emit a single snapshot —
  the animation just won't have intermediate frames.

## Step 3 — Write an advice function

Advice is what makes this a *teaching* tool. Follow the style of `_generate_advice` /
`_generate_advice_knn`: compare train vs test accuracy, name the failure mode, suggest a fix.

```python
def _generate_advice_random_forest(train_acc, test_acc, n_estimators):
    advice = []
    gap = train_acc - test_acc
    if gap > 0.15:
        advice.append({"type": "warning", "text":
            f"Overfitting: {train_acc:.0%} train vs {test_acc:.0%} test. More trees won't fix a "
            f"deep-tree overfit — try limiting max_depth."})
    else:
        advice.append({"type": "success", "text":
            f"Good balance: {test_acc:.0%} test accuracy with only {gap:.0%} gap. Averaging "
            f"{n_estimators} trees smooths out the noise any single tree would chase."})
    return advice
```

## Step 4 — Register the trainer

Add your function to the `TRAINERS` dispatch dict:

```python
TRAINERS = {
    "decision_tree": _train_decision_tree,
    "knn": _train_knn,
    "random_forest": _train_random_forest,   # <-- your id must match the frontend card id
}
```

Also import the estimator at the top of the file, e.g.
`from sklearn.ensemble import RandomForestClassifier`.

## Step 5 — Unlock the card on the frontend

In `frontend/app.js`, find `ALGO_TIERS` and flip your algorithm's `locked` flag to `false`
(the `id` here **must** equal the key you used in `TRAINERS`):

```js
{ id: "random_forest", name: "Random Forest", locked: false,
  blurb: "Combines many decision trees and lets them vote together for a sturdier answer." },
```

## Step 6 — Add hyperparameter controls (if needed)

1. In `frontend/index.html`, add a `<div id="rfControls">` block near `dtControls`/`knnControls`
   with your sliders/selects.
2. In `app.js` → `applyAlgorithmUI(algorithm)`, toggle its visibility:
   ```js
   document.getElementById("rfControls").style.display =
     algorithm === "random_forest" ? "block" : "none";
   ```
3. In `runTraining()`, include your field in the POST `payload` so it reaches the backend.
4. If it has a live-value label, register it with `bindRangeDisplay(...)` in `init()`.

## Step 7 — (Optional) Custom secondary visualization

The right-hand panel shows a **tree diagram** (decision tree) or a **neighbor radar** (KNN). If your
algorithm has a natural visual, add a branch in `applyAlgorithmUI` and a render path in
`runTraining`. If not, that's fine — the decision-boundary panel is always shown.

---

## Testing your algorithm

Run the server (`uvicorn main:app --reload --port 8000`) and hit the endpoint directly:

```bash
curl -s -X POST http://127.0.0.1:8000/api/train \
  -H "Content-Type: application/json" \
  -d '{"dataset_id":"iris_2f","algorithm":"random_forest","n_estimators":20,"player_name":"DEV"}' \
  | python -m json.tool | head -40
```

Check that: the response has `snapshots` with `boundary` grids, `final.test_accuracy` looks sane,
and `advice` is populated. Then open the app in the browser and click through all 4 pages.

> Tip: training writes a real entry to `backend/leaderboard.json`. Remove your `"DEV"`/test entries
> before committing (or `git checkout backend/leaderboard.json`).

---

## Pull request checklist

- [ ] Trainer registered in `TRAINERS`; estimator imported.
- [ ] Every snapshot includes `step_label`, `train_accuracy`, `test_accuracy`, and a valid `boundary`.
- [ ] Hyperparameters are **clamped** to a safe range.
- [ ] An advice function explains at least the overfit / good-fit cases.
- [ ] Frontend card `id` matches the `TRAINERS` key and is unlocked.
- [ ] New controls (if any) are wired in `applyAlgorithmUI` **and** the POST payload.
- [ ] Verified end to end in the browser across all 4 pages.
- [ ] No stray test entries left in `backend/leaderboard.json`; no `__pycache__` / `.DS_Store` staged.

---

## Code style

- Match the existing file's style: small helper functions, `_leading_underscore` for internals,
  f-strings for the human-facing advice text.
- Keep the backend a single file (`main.py`) unless there's a strong reason to split it.
- Frontend is intentionally framework-free — stick to vanilla JS + Canvas/SVG.

Happy hacking! 🚀
