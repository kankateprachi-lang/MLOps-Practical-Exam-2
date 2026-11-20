from pathlib import Path
import sys

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    digits = load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Handwritten-Digit-Classification")

    models = {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, random_state=42),
        ),
        "DecisionTree": DecisionTreeClassifier(max_depth=15, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=150, random_state=42
        ),
    }

    best_accuracy = -1
    best_model = None
    best_name = ""

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            weighted_f1 = f1_score(y_test, predictions, average="weighted")

            mlflow.log_param("model_name", name)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("weighted_f1", weighted_f1)
            mlflow.sklearn.log_model(
                model,
                name="model",
                skops_trusted_types=SKOPS_TRUSTED_TYPES,
            )

            print(
                name,
                "accuracy:",
                round(accuracy, 4),
                "weighted F1:",
                round(weighted_f1, 4),
            )

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_name = name

    if best_model is None:
        raise RuntimeError("No model was trained.")

    model_path = Path("models") / "best_model"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    mlflow.sklearn.save_model(
        best_model,
        str(model_path),
        skops_trusted_types=SKOPS_TRUSTED_TYPES,
    )
    print("Best model:", best_name, "| Accuracy:", round(best_accuracy, 4))
    print(f"Model saved in {model_path}")


if __name__ == "__main__":
    main()
