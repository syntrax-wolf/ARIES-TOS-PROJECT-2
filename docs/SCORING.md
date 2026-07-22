# The ARIES Score — how runs are ranked

> **Goal (from the brief):** *the best combination of algorithm + hyperparameters for a given dataset
> must land highest on the leaderboard.* Not the biggest model. Not the luckiest one. The one that
> actually generalizes.

---

## 1. Why not R², and why not plain accuracy

**R² is out.** R² measures how much of the *variance of a continuous target* a regression explains.
Our targets are class labels (`setosa` / `versicolor` / `virginica`) — there is no variance to explain
and no meaningful residual. `sklearn.metrics.r2_score` on class indices returns a number, but that
number is nonsense: it would rate "predicted class 2 instead of class 1" as a smaller error than
"predicted class 3 instead of class 1", even though the labels have no order.

**Plain accuracy is out as the *primary* metric.** `cancer_2f` is 357 benign vs 212 malignant. A model
that predicts "benign" for everything scores 63% accuracy while catching zero tumors. On a leaderboard
that is a free 630 points for a model that does nothing. Any dataset we add later makes this worse.

## 2. The primary metric: macro-F1 on the held-out test set

**`test_macro_f1`** — F1 computed per class, then averaged unweighted.

- Every class carries equal weight, so ignoring a minority class is punished immediately.
- It balances precision and recall, so neither over-predicting nor under-predicting a class is free.
- It is a single number in `[0, 1]`, directly comparable across algorithms and hyperparameters —
  which is exactly what "best combination for this dataset" requires.

The split is **fixed and identical for every student**: stratified, `test_size=0.30`,
`random_state=42`. Nobody wins by getting an easier split. The test set is never used for fitting,
so `test_macro_f1` is a genuine generalization estimate.

## 3. The final score

```
gap              = train_macro_f1 - test_macro_f1
overfit_penalty  = 0.50 * max(0, gap - 0.03)
simplicity_bonus = 0.02 * (1 - complexity_ratio)

ARIES Score      = round( 1000 * clamp( test_macro_f1 - overfit_penalty + simplicity_bonus, 0, 1 ) )
```

An integer **0–1000**. Arcade-legible, and every term is explainable to a first-year student in one
sentence.

| Term | What it does | Why |
|------|--------------|-----|
| `test_macro_f1` | Carries ~98% of the score. | Generalization is the whole game. |
| `overfit_penalty` | Subtracts half of every point of train/test gap beyond 3%. | This is the term that makes a *tuned* model beat a *maxed-out* model. A depth-20 tree at 100% train / 88% test loses ~45 points to a depth-4 tree at 94% train / 92% test. |
| `0.03` tolerance | A noise band. | On a 45-row test set one flipped sample moves the gap ~2%. Students shouldn't be punished for sampling noise. |
| `simplicity_bonus` | Up to +20 points for the leaner model. | Pure tie-break. Two models at identical F1: the simpler one wins. Occam, gamified. |

`complexity_ratio` is the model's effective size normalized against the algorithm's configured
maximum — leaf count for a single tree, `n_estimators × max_depth` for an ensemble — clamped to
`[0, 1]`. It is deliberately capped at 2% of the score so it can break ties but can never outrank real
predictive performance.

**Ordering on the leaderboard:** `aries_score DESC`, then `test_macro_f1 DESC`, then `gap ASC`, then
`created_at ASC` (earliest submission wins a true tie).

## 4. Overfit verdicts — what drives the roasts

The same `gap` that costs points also picks the message. Verdicts are stored on every run so the
comment engine is pure lookup, and new roast lines can be added in the DB without a redeploy.

| Verdict | Condition | Tone |
|---------|-----------|------|
| `underfit` | `test_macro_f1 < 0.60` and `gap <= 0.03` | "Your model gave up before it started." |
| `clean` | `gap <= 0.03` | "Textbook. Train and test agree — this thing actually learned." |
| `slight_overfit` | `0.03 < gap <= 0.10` | "Nice pick, pal. Slightly memorizing, but you got away with it." |
| `overfit` | `0.10 < gap <= 0.20` | "Look at you overfitting. It knows the training set by heart and panics on anything new." |
| `hard_overfit` | `gap > 0.20` | "That's not learning, that's a lookup table with extra steps." |

Evaluation order is top to bottom — `underfit` is checked before `clean` so a model that agrees with
itself at 40% F1 gets called out instead of praised.

## 5. Stored but *not* ranked: stability

Every run also records **5-fold stratified CV macro-F1** (mean and std) over the training portion.
It does not enter the score — it powers a "stable / lucky" badge and lets us tell a student their
great test score came from one fortunate split. Cheap to compute: these datasets are 150–569 rows and
tree models fit in milliseconds.

## 6. Reference implementation

```python
from sklearn.metrics import f1_score

OVERFIT_TOLERANCE  = 0.03
OVERFIT_WEIGHT     = 0.50
SIMPLICITY_WEIGHT  = 0.02

def aries_score(train_f1: float, test_f1: float, complexity_ratio: float) -> int:
    gap     = train_f1 - test_f1
    penalty = OVERFIT_WEIGHT * max(0.0, gap - OVERFIT_TOLERANCE)
    bonus   = SIMPLICITY_WEIGHT * (1.0 - min(max(complexity_ratio, 0.0), 1.0))
    raw     = test_f1 - penalty + bonus
    return round(1000 * min(max(raw, 0.0), 1.0))

def overfit_verdict(train_f1: float, test_f1: float) -> str:
    gap = train_f1 - test_f1
    if test_f1 < 0.60 and gap <= 0.03: return "underfit"
    if gap <= 0.03:  return "clean"
    if gap <= 0.10:  return "slight_overfit"
    if gap <= 0.20:  return "overfit"
    return "hard_overfit"
```

Macro-F1 itself: `f1_score(y_true, y_pred, average="macro", zero_division=0)`.

## 7. Worked example — why the penalty matters

`iris_2f`, Decision Tree, two students:

| | Student A | Student B |
|---|---|---|
| Hyperparameters | `max_depth=20, min_samples_split=2` | `max_depth=3, min_samples_split=6` |
| Train macro-F1 | 1.000 | 0.962 |
| Test macro-F1 | 0.911 | 0.933 |
| gap | 0.089 | 0.029 |
| overfit penalty | `0.5 × (0.089 − 0.03)` = 0.030 | 0 (inside tolerance) |
| simplicity bonus | ≈ 0.001 | ≈ 0.017 |
| **ARIES Score** | **882** | **950** |

Student B wins by 68 points — and the app can explain *exactly* which 68. Under raw accuracy the two
would have been within a couple of points of each other, and the lesson would have been lost.
