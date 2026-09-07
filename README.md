# ARIES // ML Training Playground

An interactive, arcade-style web app for **learning how classification models work** — pick a
dataset, tune a model, watch it train step by step, read plain-English coaching on
overfitting/underfitting, then compete on a high-score leaderboard.

> Built for **ARIES — Artificial Intelligence Society, IIT Delhi**.

The core idea: *machine learning for classification is just the search for a function whose curve
best separates one class from another.* Every screen is designed to make that search **visible** —
decision boundaries, tree structures, and nearest-neighbor "radar" sweeps are all drawn live.

---

## Features

- **4-step guided flow** — Learn → Setup → Train → Results.
- **Live training visualization** — the decision boundary repaints at every step (tree depth `1→N`,
  or KNN `k=1→N`), alongside the tree diagram or a KNN neighbor-radar scan of the test set.
- **Explainable advice engine** — detects overfitting, underfitting, and good generalization and
  explains *why* in beginner-friendly language.
- **Interactive results** — drop query points on the final boundary and watch them get classified.
- **Arcade leaderboard** — top test-accuracy runs are saved per dataset + algorithm.

### Datasets (each reduced to 2 features so the boundary draws in 2D)

| id          | Data                          | Features                        | Classes |
|-------------|-------------------------------|---------------------------------|---------|
| `iris_2f`   | Iris flowers                  | petal length, petal width       | 3       |
| `wine_2f`   | Wine chemistry                | alcohol, flavanoids             | 3       |
| `cancer_2f` | Breast tumors                 | mean radius, mean texture       | 2       |

### Algorithms

| Algorithm            | Status         |
|----------------------|----------------|
| K-Nearest Neighbors  | ✅ Implemented |
| Decision Tree        | ✅ Implemented |
| Random Forest        | 🔒 Planned     |
| AdaBoost             | 🔒 Planned     |
| Logistic / Poly Reg. | 🔒 Planned     |
| Support Vector Machine | 🔒 Planned   |
| Neural Network       | 🔒 Planned     |

Want to implement one of the planned algorithms? See **[CONTRIBUTING.md](CONTRIBUTING.md)** — it
walks you through adding a new algorithm end to end.

---

## Tech stack

- **Backend:** [FastAPI](https://fastapi.tiangolo.com/) + [scikit-learn](https://scikit-learn.org/)
  + NumPy, served by Uvicorn. Single file: `backend/main.py`.
- **Frontend:** vanilla HTML / CSS / JavaScript (no build step, no framework) using Canvas 2D + SVG.
- **Storage:** a flat `backend/leaderboard.json` file (no database).

---

## Project layout

```
ARIES-TOS-PROJECT-2/
├── backend/
│   ├── main.py            # FastAPI app: datasets, training, advice, leaderboard, static mount
│   ├── leaderboard.json   # persisted high scores (flat file)
│   └── requirements.txt
├── frontend/
│   ├── index.html         # 4-page wizard markup
│   ├── app.js             # all UI logic + canvas/SVG rendering
│   └── style.css
├── README.md
└── CONTRIBUTING.md        # how to add a new algorithm
```

---

## Getting started

> This project is developed against **Anaconda Python** (Python 3.13). Use a virtual environment so
> you don't install into your base/system Python.

### 1. Install dependencies

```bash
cd backend

# Recommended: an isolated conda environment
conda create -n aries python=3.11 -y
conda activate aries

pip install -r requirements.txt
```

`requirements.txt`:

```
fastapi>=0.110
uvicorn[standard]>=0.29
scikit-learn>=1.4
numpy>=1.26
pydantic>=2.6
```

### 2. Run the server

```bash
# from the backend/ directory
uvicorn main:app --reload --port 8000
```

FastAPI serves **both** the API and the static frontend, so just open:

```
http://127.0.0.1:8000/
```

---

## API reference

Base path: `/api`

| Method | Endpoint             | Purpose                                                        |
|--------|----------------------|----------------------------------------------------------------|
| `GET`  | `/api/datasets`      | List available datasets (id, description, features, classes).  |
| `GET`  | `/api/dataset_preview?dataset_id=` | Train/test points for the scatter preview.       |
| `POST` | `/api/train`         | Train a model, return per-step snapshots + advice + rank.      |
| `GET`  | `/api/leaderboard?dataset_id=&algorithm=` | Top 10 scores (filters optional).         |

### `POST /api/train`

Request body (`TrainRequest`):

```json
{
  "dataset_id": "iris_2f",
  "algorithm": "decision_tree",
  "max_depth": 3,
  "min_samples_split": 2,
  "criterion": "gini",
  "n_neighbors": 5,
  "weights": "uniform",
  "player_name": "AAA"
}
```

Response (abridged):

```json
{
  "algorithm": "decision_tree",
  "feature_names": ["petal length (cm)", "petal width (cm)"],
  "class_names": ["setosa", "versicolor", "virginica"],
  "points": { "train": [...], "test": [...] },
  "snapshots": [ { "step_label": "Depth 1", "boundary": {...}, "tree": {...}, "...": "..." } ],
  "final": { "...": "the last snapshot" },
  "advice": [ { "type": "success", "text": "..." } ],
  "entry": { "...": "leaderboard entry" },
  "leaderboard_rank": 3,
  "leaderboard_total": 17
}
```

Each `snapshot.boundary` is a coarse grid of predicted class indices (`resolution × resolution`) that
the frontend paints as the colored decision surface.

---

## Development notes

- The backend intentionally trains a **range** of models per request (every depth `1..N`, every
  `k` `1..N`) so the frontend can animate the model "getting smarter" one step at a time.
- Feature ranges are padded ~8% so points aren't drawn on the canvas edge.
- The leaderboard keeps the top 50 entries on disk and returns the top 10 per query.

---

## License

Educational project by ARIES, IIT Delhi. See the repository for license details.
