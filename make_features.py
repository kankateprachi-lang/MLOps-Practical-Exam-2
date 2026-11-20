from pathlib import Path

import pandas as pd
from sklearn.datasets import load_digits


ROOT = Path(__file__).resolve().parent


def main():
    digits = load_digits()
    features = pd.DataFrame(
        digits.data.astype("float32"),
        columns=[f"pixel_{index}" for index in range(digits.data.shape[1])],
    )
    features["digit_id"] = range(len(features))
    features["target"] = digits.target.astype("int64")
    features["event_timestamp"] = pd.Timestamp.now(tz="UTC").floor("s")

    output_path = ROOT / "feature_repo" / "data" / "digit_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    print(f"Feature data saved: {output_path} ({features.shape})")


if __name__ == "__main__":
    main()
