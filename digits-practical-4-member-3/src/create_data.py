from pathlib import Path

import pandas as pd
from sklearn.datasets import load_digits


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    digits = load_digits()
    df = pd.DataFrame(digits.data, columns=[f"pixel_{index}" for index in range(64)])
    df["target"] = digits.target

    output_path = ROOT / "data" / "raw" / "digits.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Raw dataset created: {output_path} ({df.shape})")


if __name__ == "__main__":
    main()
