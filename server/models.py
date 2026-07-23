"""
Canonical (referee) models.

The browser trains its own TypeScript models to drive the animation. Those are
teaching implementations, not the scoreboard: a student can edit the numbers
they POST, so a client-reported score can never be ranked.

Instead the server retrains from scratch, using scikit-learn, on the identical
train/test split rebuilt from the seed (see dataset.py). Every student on a
given seed is therefore judged by the same referee under the same conditions.

The server's score will not match the browser's exactly — different
implementations of the same algorithm. That is expected and is surfaced to the
student rather than hidden.
"""
from dataclasses import dataclass

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from dataset import build_dataset

RANDOM_STATE = 42

# Mirrors src/lib/gradientBoosting.ts
LEARNER_MAX_DEPTH = 2

# Algorithms that are ranked. "cnn" is deliberately absent: it ships pre-trained
# weights, so there is nothing the student tuned to compare (UNRATED_ALGORITHMS
# in src/state/store.ts).
RATED_ALGORITHMS = {
    "decision-tree",
    "random-forest",
    "gradient-boosting",
    "knn",
    "neural-net",
}

# Bounds copied from the sliders in src/pages/Hyperparameters.tsx. Values are
# clamped, never rejected — the UI cannot produce anything outside these, so an
# out-of-range value means someone is poking the API by hand.
BOUNDS = {
    "maxDepth": (1, 15),
    "numTrees": (3, 9),
    "hiddenLayers": (1, 3),
    "nodesPerLayer": (4, 16),
    "k": (1, 15),
    "numLearners": (3, 16),
}

METRICS = {"euclidean", "manhattan"}


@dataclass
class Score:
    accuracy: float
    macro_f1: float
    train_accuracy: float
    n_train: int
    n_test: int


def clamp_hyperparams(raw: dict) -> dict:
    """Clamp to the slider ranges and drop anything unrecognised."""
    out = {}
    for name, (lo, hi) in BOUNDS.items():
        value = raw.get(name)
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = lo
        out[name] = max(lo, min(value, hi))

    metric = raw.get("metric")
    out["metric"] = metric if metric in METRICS else "euclidean"
    return out


def _build_estimator(algorithm: str, hp: dict):
    if algorithm == "decision-tree":
        return DecisionTreeClassifier(max_depth=hp["maxDepth"], random_state=RANDOM_STATE)

    if algorithm == "random-forest":
        return RandomForestClassifier(
            n_estimators=hp["numTrees"],
            max_depth=hp["maxDepth"],
            random_state=RANDOM_STATE,
            n_jobs=1,
        )

    if algorithm == "gradient-boosting":
        return GradientBoostingClassifier(
            n_estimators=hp["numLearners"],
            max_depth=LEARNER_MAX_DEPTH,
            random_state=RANDOM_STATE,
        )

    if algorithm == "knn":
        return KNeighborsClassifier(n_neighbors=hp["k"], metric=hp["metric"])

    if algorithm == "neural-net":
        return MLPClassifier(
            hidden_layer_sizes=tuple([hp["nodesPerLayer"]] * hp["hiddenLayers"]),
            random_state=RANDOM_STATE,
            max_iter=400,
        )

    raise ValueError(f"Unknown algorithm '{algorithm}'.")


def evaluate(algorithm: str, hyperparams: dict, seed: int) -> Score:
    """Rebuild the student's split, train the canonical model, score it."""
    X_train, y_train, X_test, y_test = build_dataset(seed)

    model = _build_estimator(algorithm, hyperparams)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    return Score(
        accuracy=float(accuracy_score(y_test, predictions)),
        macro_f1=float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        train_accuracy=float(model.score(X_train, y_train)),
        n_train=int(len(y_train)),
        n_test=int(len(y_test)),
    )


def summarize(algorithm: str, hp: dict) -> str:
    """Short human-readable hyperparameter summary for the leaderboard row."""
    if algorithm == "random-forest":
        return f"{hp['numTrees']} trees, max depth {hp['maxDepth']}"
    if algorithm == "neural-net":
        layers = hp["hiddenLayers"]
        return f"{layers} hidden layer{'s' if layers > 1 else ''}, {hp['nodesPerLayer']} nodes each"
    if algorithm == "knn":
        return f"k={hp['k']}, {hp['metric']}"
    if algorithm == "gradient-boosting":
        return f"{hp['numLearners']} learners"
    return f"max depth {hp['maxDepth']}"


ALGORITHM_LABELS = {
    "decision-tree": "Decision Tree",
    "random-forest": "Random Forest",
    "gradient-boosting": "Gradient Boosting",
    "knn": "k-Nearest Neighbors",
    "neural-net": "Neural Network",
}
