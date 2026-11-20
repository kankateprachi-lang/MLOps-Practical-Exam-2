from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    input_path = ROOT / "data" / "processed" / "digits_processed.csv"
    df = pd.read_csv(input_path)
    pixel_cols = [column for column in df.columns if column.startswith("pixel_")]

    if len(df) != 1797:
        raise ValueError(f"Expected 1797 rows, found {len(df)}")
    if len(pixel_cols) != 64:
        raise ValueError(f"Expected 64 pixel features, found {len(pixel_cols)}")
    if not df["target"].between(0, 9).all():
        raise ValueError("Target values must be between 0 and 9")
    if df.isnull().any().any():
        raise ValueError("Missing values found")
    if df[pixel_cols].min().min() < 0 or df[pixel_cols].max().max() > 1:
        raise ValueError("Pixel values must be in the range [0, 1]")

    print("Validation successful")
    print(f"Rows: {len(df)} | Features: {len(pixel_cols)}")


if __name__ == "__main__":
    main()
