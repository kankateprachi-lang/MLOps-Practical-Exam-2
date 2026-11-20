from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    input_path = ROOT / "data" / "raw" / "digits.csv"
    output_path = ROOT / "data" / "processed" / "digits_processed.csv"

    df = pd.read_csv(input_path)
    pixel_cols = [column for column in df.columns if column.startswith("pixel_")]
    df[pixel_cols] = df[pixel_cols] / 16.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Processed dataset saved: {output_path} ({df.shape})")
    print(
        "Pixel value range:",
        df[pixel_cols].min().min(),
        "to",
        df[pixel_cols].max().max(),
    )


if __name__ == "__main__":
    main()
