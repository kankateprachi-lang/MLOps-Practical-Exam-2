from pathlib import Path

import pandas as pd
from sklearn.datasets import load_digits


def main():
    digits = load_digits()
    df = pd.DataFrame(digits.data, columns=[f"pixel_{i}" for i in range(64)])
    df["target"] = digits.target

    output_path = Path("data") / "digits.csv"
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Dataset saved: {output_path}")
    print("Shape:", df.shape)
    print(df.head())


if __name__ == "__main__":
    main()
