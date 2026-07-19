"""
ARIES ML Training Playground - backend
Serves the static frontend and a small API for training + visualizing
ML models step-by-step, plus an arcade-style leaderboard.
"""
import json
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
LEADERBOARD_FILE = BASE_DIR / "leaderboard.json"

app = FastAPI(title="ARIES ML Playground")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Dataset registry — each dataset is reduced to its 2 most informative
# features so every algorithm's decision surface can be drawn directly in 2D.
# ---------------------------------------------------------------------------

def _make_dataset(loader, feature_indices, name, description):
    raw = loader()
    X = raw.data[:, feature_indices]
    y = raw.target
    feature_names = [raw.feature_names[i] for i in feature_indices]
    class_names = list(raw.target_names)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return {
        "id": name,
        "description": description,
        "feature_names": feature_names,
        "class_names": class_names,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "n_samples": len(X),
    }


DATASETS = {
    "iris_2f": _make_dataset(
        load_iris, [2, 3], "iris_2f",
        "Iris flowers — petal length vs petal width (3 species)",
    ),
    "wine_2f": _make_dataset(
        load_wine, [0, 6], "wine_2f",
        "Wine chemistry — alcohol vs flavanoids (3 cultivars)",
    ),
    "cancer_2f": _make_dataset(
        load_breast_cancer, [0, 1], "cancer_2f",
        "Breast tumors — mean radius vs mean texture (benign/malignant)",
    ),
}


@app.get("/api/datasets")
def list_datasets():
    return [
        {
            "id": d["id"],
            "description": d["description"],
            "feature_names": d["feature_names"],
            "class_names": d["class_names"],
            "n_samples": d["n_samples"],
        }
        for d in DATASETS.values()
    ]


def _points_payload(dataset):
    class_names = dataset["class_names"]
    return {
        "train": [
            {"x": float(p[0]), "y": float(p[1]), "label": class_names[int(l)]}
            for p, l in zip(dataset["X_train"], dataset["y_train"])
        ],
        "test": [
            {"x": float(p[0]), "y": float(p[1]), "label": class_names[int(l)]}
            for p, l in zip(dataset["X_test"], dataset["y_test"])
        ],
    }


@app.get("/api/dataset_preview")
def dataset_preview(dataset_id: str):
    dataset = DATASETS[dataset_id]
    return {
        "feature_names": dataset["feature_names"],
        "class_names": dataset["class_names"],
        "points": _points_payload(dataset),
    }


# ---------------------------------------------------------------------------
# Tree -> JSON (for D3 hierarchy rendering)
# ---------------------------------------------------------------------------

def _tree_to_dict(tree, node_id, feature_names, class_names):
    left = tree.children_left[node_id]
    right = tree.children_right[node_id]
    value = tree.value[node_id][0]
    prediction = class_names[int(np.argmax(value))]
    is_leaf = left == -1 and right == -1

    node = {
        "id": int(node_id),
        "samples": int(tree.n_node_samples[node_id]),
        "impurity": round(float(tree.impurity[node_id]), 4),
        "value": [int(v) for v in value],
        "prediction": prediction,
        "is_leaf": bool(is_leaf),
    }
    if not is_leaf:
        node["feature"] = feature_names[tree.feature[node_id]]
        node["threshold"] = round(float(tree.threshold[node_id]), 3)
        node["children"] = [
            _tree_to_dict(tree, left, feature_names, class_names),
            _tree_to_dict(tree, right, feature_names, class_names),
        ]
    return node


def _decision_boundary_grid(model, x_range, y_range, resolution=60):
    xx, yy = np.meshgrid(
        np.linspace(x_range[0], x_range[1], resolution),
        np.linspace(y_range[0], y_range[1], resolution),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    preds = model.predict(grid_points).reshape(xx.shape)
    return {
        "x_min": float(x_range[0]), "x_max": float(x_range[1]),
        "y_min": float(y_range[0]), "y_max": float(y_range[1]),
        "resolution": resolution,
        "grid": preds.astype(int).tolist(),
    }


def _feature_range(dataset, feature_idx, pad_ratio=0.08):
    col = np.concatenate([dataset["X_train"][:, feature_idx], dataset["X_test"][:, feature_idx]])
    lo, hi = col.min(), col.max()
    pad = (hi - lo) * pad_ratio
    return (lo - pad, hi + pad)


# ---------------------------------------------------------------------------
# Advice engine — simple, explainable rules a beginner can learn from.
# ---------------------------------------------------------------------------

def _generate_advice(train_acc, test_acc, depth, max_depth_requested, n_leaves, n_train):
    advice = []
    gap = train_acc - test_acc

    if gap > 0.15:
        advice.append({
            "type": "warning",
            "text": (
                f"Overfitting detected: your tree scored {train_acc:.0%} on training data but only "
                f"{test_acc:.0%} on unseen test data. It memorized noise instead of learning the real "
                f"pattern. Try lowering max_depth or raising min_samples_split."
            ),
        })
    elif test_acc < 0.7:
        advice.append({
            "type": "warning",
            "text": (
                f"Underfitting: at depth {depth} the tree is too simple to separate the classes well "
                f"(only {test_acc:.0%} test accuracy). Try increasing max_depth."
            ),
        })
    else:
        advice.append({
            "type": "success",
            "text": (
                f"Good balance: {test_acc:.0%} test accuracy with only {gap:.0%} gap from training "
                f"accuracy — the tree generalizes well instead of memorizing."
            ),
        })

    if n_leaves > n_train * 0.3:
        advice.append({
            "type": "info",
            "text": (
                f"This tree has {n_leaves} leaves for only {n_train} training points — many leaves "
                f"cover just 1-2 samples, a classic overfitting fingerprint."
            ),
        })

    if depth == max_depth_requested and max_depth_requested >= 3:
        advice.append({
            "type": "info",
            "text": "Notice how each extra level only carves the space into smaller rectangles — "
                     "diminishing returns set in fast after a few splits.",
        })

    return advice


def _generate_advice_knn(train_acc, test_acc, k, n_train, weights):
    advice = []
    gap = train_acc - test_acc

    if k == 1:
        advice.append({
            "type": "warning",
            "text": (
                "k=1 means every prediction copies whichever single training point happens to be "
                "closest — the boundary gets jagged and chases noise. Try a slightly larger k."
            ),
        })
    elif gap > 0.15:
        advice.append({
            "type": "warning",
            "text": (
                f"Overfitting detected: {train_acc:.0%} training accuracy vs {test_acc:.0%} test "
                f"accuracy at k={k}. Try increasing k to smooth the boundary."
            ),
        })
    elif k > n_train * 0.25:
        advice.append({
            "type": "warning",
            "text": (
                f"k={k} is large relative to your {n_train} training points — the model is now "
                f"mostly voting with the overall class balance instead of local neighborhoods "
                f"(underfitting). Try a smaller k."
            ),
        })
    elif test_acc < 0.7:
        advice.append({
            "type": "warning",
            "text": f"Only {test_acc:.0%} test accuracy at k={k} — try a different k or dataset.",
        })
    else:
        advice.append({
            "type": "success",
            "text": (
                f"Good balance: {test_acc:.0%} test accuracy at k={k} with only {gap:.0%} gap "
                f"between train and test — the neighborhood size fits the data's structure well."
            ),
        })

    if weights == "distance":
        advice.append({
            "type": "info",
            "text": "Distance-weighting means closer neighbors get a louder vote than farther ones "
                     "inside the same k — watch how it softens the boundary near class borders.",
        })

    return advice


# ---------------------------------------------------------------------------
# Leaderboard (arcade high-score style)
# ---------------------------------------------------------------------------

def _load_leaderboard():
    if LEADERBOARD_FILE.exists():
        return json.loads(LEADERBOARD_FILE.read_text())
    return []


def _save_leaderboard(entries):
    LEADERBOARD_FILE.write_text(json.dumps(entries, indent=2))


class TrainRequest(BaseModel):
    dataset_id: str
    algorithm: str = "decision_tree"
    max_depth: int = 3
    min_samples_split: int = 2
    criterion: str = "gini"
    n_neighbors: int = 5
    weights: str = "uniform"
    player_name: str = "AAA"


def _train_decision_tree(req, dataset, x_range, y_range):
    feature_names, class_names = dataset["feature_names"], dataset["class_names"]
    X_train, X_test = dataset["X_train"], dataset["X_test"]
    y_train, y_test = dataset["y_train"], dataset["y_test"]

    max_depth = max(1, min(req.max_depth, 8))
    snapshots = []

    for depth in range(1, max_depth + 1):
        model = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_split=max(2, req.min_samples_split),
            criterion=req.criterion,
            random_state=42,
        )
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        snapshots.append({
            "step_label": f"Depth {depth}",
            "depth": depth,
            "train_accuracy": round(train_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "n_leaves": int(model.get_n_leaves()),
            "tree": _tree_to_dict(model.tree_, 0, feature_names, class_names),
            "boundary": _decision_boundary_grid(model, x_range, y_range),
        })

    final = snapshots[-1]
    advice = _generate_advice(
        final["train_accuracy"], final["test_accuracy"], final["depth"],
        max_depth, final["n_leaves"], len(X_train),
    )
    params_summary = f"depth={max_depth}"
    return snapshots, final, advice, params_summary


def _train_knn(req, dataset, x_range, y_range):
    X_train, X_test = dataset["X_train"], dataset["X_test"]
    y_train, y_test = dataset["y_train"], dataset["y_test"]

    k_requested = max(1, req.n_neighbors)
    k_cap = max(1, min(k_requested, len(X_train) - 1, 25))
    weights = req.weights if req.weights in ("uniform", "distance") else "uniform"
    snapshots = []

    for k in range(1, k_cap + 1):
        model = KNeighborsClassifier(n_neighbors=k, weights=weights)
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        snapshots.append({
            "step_label": f"k = {k}",
            "k": k,
            "weights": weights,
            "train_accuracy": round(train_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "boundary": _decision_boundary_grid(model, x_range, y_range),
        })

    final = snapshots[-1]
    advice = _generate_advice_knn(
        final["train_accuracy"], final["test_accuracy"], final["k"], len(X_train), weights,
    )
    params_summary = f"k={k_cap}, {weights}"
    return snapshots, final, advice, params_summary


TRAINERS = {
    "decision_tree": _train_decision_tree,
    "knn": _train_knn,
}


@app.post("/api/train")
def train(req: TrainRequest):
    dataset = DATASETS[req.dataset_id]

    x_range = _feature_range(dataset, 0)
    y_range = _feature_range(dataset, 1)

    trainer = TRAINERS[req.algorithm]
    snapshots, final, advice, params_summary = trainer(req, dataset, x_range, y_range)

    entry = {
        "player_name": req.player_name[:8] or "AAA",
        "dataset_id": req.dataset_id,
        "algorithm": req.algorithm,
        "params_summary": params_summary,
        "test_accuracy": final["test_accuracy"],
        "timestamp": time.time(),
    }
    board = _load_leaderboard()
    board.append(entry)
    board.sort(key=lambda e: e["test_accuracy"], reverse=True)
    board = board[:50]
    _save_leaderboard(board)

    return {
        "algorithm": req.algorithm,
        "feature_names": dataset["feature_names"],
        "class_names": dataset["class_names"],
        "points": _points_payload(dataset),
        "snapshots": snapshots,
        "final": final,
        "advice": advice,
        "entry": entry,
        "leaderboard_rank": board.index(entry) + 1 if entry in board else None,
        "leaderboard_total": len(board),
    }


@app.get("/api/leaderboard")
def leaderboard(dataset_id: str | None = None, algorithm: str | None = None):
    board = _load_leaderboard()
    if dataset_id:
        board = [e for e in board if e["dataset_id"] == dataset_id]
    if algorithm:
        board = [e for e in board if e["algorithm"] == algorithm]
    return board[:10]


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
