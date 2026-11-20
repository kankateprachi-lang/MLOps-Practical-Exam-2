from pathlib import Path

import joblib
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    digits = load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, random_state=42),
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Dataset shape:", X.shape)
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print(classification_report(y_test, predictions))

    model_path = Path("models") / "digit_classifier.joblib"
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved in {model_path}")


if __name__ == "__main__":
    main()
