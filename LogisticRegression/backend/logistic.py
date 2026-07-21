import numpy as np


class LogisticRegression:

    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        # Prevent overflow in exp()
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):

        m, n = X.shape

        self.w = np.zeros(n)
        self.b = 0

        history = []

        previous_loss = float("inf")

        for epoch in range(self.epochs):

            # ----------------------------
            # Forward Pass
            # ----------------------------
            z = np.dot(X, self.w) + self.b

            probabilities = self.sigmoid(z)

            # Binary predictions
            predictions = (probabilities >= 0.5).astype(int)

            # ----------------------------
            # Metrics
            # ----------------------------
            loss = -np.mean(
                y * np.log(probabilities + 1e-9)
                + (1 - y) * np.log(1 - probabilities + 1e-9)
            )

            accuracy = np.mean(predictions == y)

            # ----------------------------
            # Gradient Descent
            # ----------------------------
            dw = (1 / m) * np.dot(X.T, (probabilities - y))
            db = (1 / m) * np.sum(probabilities - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            # ----------------------------
            # Save History
            # ----------------------------
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss),
                    "accuracy": float(accuracy),

                    "weights": self.w.tolist(),
                    "bias": float(self.b),

                    "predictions": predictions.tolist(),
                    "probabilities": probabilities.tolist(),
                }
            )

            # ----------------------------
            # Early Stopping
            # ----------------------------
            if abs(previous_loss - loss) < 1e-7:
                print(f"Converged at epoch {epoch}")
                break

            previous_loss = loss

        return history