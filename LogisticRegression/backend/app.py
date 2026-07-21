from flask import Flask, jsonify, request
from flask_cors import CORS

from sklearn.datasets import (
    load_iris,
    make_moons,
    make_circles,
    make_blobs,
    make_classification,
)

from logistic import LogisticRegression

app = Flask(__name__)
CORS(app)


# ==========================================
# Dataset Loader
# ==========================================

def get_dataset(name):
    name = name.lower()

    if name == "iris":
        iris = load_iris()

        X = iris.data[:, :2]
        y = (iris.target == 0).astype(int)

    elif name == "moons":
        X, y = make_moons(
            n_samples=200,
            noise=0.15,
            random_state=42
        )

    elif name == "circles":
        X, y = make_circles(
            n_samples=200,
            noise=0.08,
            factor=0.5,
            random_state=42
        )

    elif name == "blobs":
        X, y = make_blobs(
            n_samples=200,
            centers=2,
            cluster_std=1.2,
            random_state=42
        )

    elif name == "linear":
        X, y = make_classification(
            n_samples=200,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            class_sep=2.0,
            random_state=42
        )

    else:
        iris = load_iris()

        X = iris.data[:, :2]
        y = (iris.target == 0).astype(int)

    return X, y


@app.route("/")
def home():
    return {"message": "Backend is working!"}


# ==========================================
# Dataset Endpoint
# ==========================================

@app.route("/dataset")
def dataset():

    dataset_name = request.args.get("name", "iris")

    X, y = get_dataset(dataset_name)

    data = []

    for i in range(len(X)):
        data.append({
            "x": float(X[i][0]),
            "y": float(X[i][1]),
            "class": int(y[i])
        })

    return jsonify(data)


# ==========================================
# Train Endpoint
# ==========================================

@app.route("/train", methods=["POST"])
def train():

    body = request.json

    dataset_name = body.get("dataset", "iris")
    learning_rate = body["learning_rate"]
    epochs = body["epochs"]

    X, y = get_dataset(dataset_name)

    model = LogisticRegression(
        lr=learning_rate,
        epochs=epochs
    )

    history = model.fit(X, y)

    return jsonify(history)


if __name__ == "__main__":
    app.run(debug=True)