# Digits Dataset Pipeline — Member 3

This standalone practical generates the Scikit-learn Digits dataset, scales its
64 pixel columns to the range 0–1, and validates the processed CSV with DVC.

## Setup

From this directory, create and activate a virtual environment on macOS, then
install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the pipeline

Run from this directory:

```bash
dvc repro
```

The stages create `data/raw/digits.csv`, create
`data/processed/digits_processed.csv`, then validate its 1,797 rows, 64 pixel
features, target labels, missing values, and scaled pixel range. A successful
run prints `Validation successful`. DVC skips stages whose dependencies have
not changed.

This folder is a DVC subproject inside the existing Git repository; the DVC
metadata and pipeline are local to this folder. Configure a shared DVC remote
separately if the generated data needs to be shared with another machine.
