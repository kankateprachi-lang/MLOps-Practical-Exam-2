# Handwritten Digit Classification — MLOps Practicals 2, 5, and 6

This repository contains three learning practicals using Scikit-learn's
handwritten Digits dataset: a baseline classifier (Practical 2), a local Feast
feature store (Practical 5), and experiment tracking with MLflow (Practical 6).
These projects are demonstrations, not OCR systems for arbitrary camera images.

## Requirements

- Python 3.11
- Git

## Set up the environment

Use Python 3.11 in a fresh virtual environment. From Git Bash, create and
activate the environment, then install the project dependencies:

```bash
python --version
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train_digits.py
```

If `python` is not recognized on Windows, use `py --version` and
`py -3.11 -m venv .venv` when creating the environment. After activation,
`python` should refer to that environment.

The Practical 2 script prints the dataset shape, test accuracy, and a
classification report for digits 0–9. It saves the model to
`models/digit_classifier.joblib`.

## Practical 5: Feast feature store

Generate a Parquet file with the 64 pixel features, digit ID, label, and UTC
event timestamp, then apply the local Feast feature definitions and materialize
the features:

```bash
python make_features.py
cd feature_repo
feast apply
feast materialize-incremental 2099-01-01T00:00:00
cd ..
python retrieve_features.py
```

The materialization end time must be later than the generated event timestamps.
The retrieval script performs a historical lookup for entity `digit_id=0` and
prints its first four pixel features. Feast's registry, SQLite online store,
and generated Parquet data are local artifacts and are excluded from Git.
Feast commands and supported configuration can vary by version; use a fresh
Python 3.11 environment and keep the installed Feast version consistent with
the feature definitions and configuration.

## Practical 6: MLflow experiment tracking

The Practical 6 script trains Logistic Regression, Decision Tree, and Random
Forest classifiers. It logs each model's test accuracy, weighted F1 score, and
model artifact to MLflow, then saves the highest-accuracy model to
`models/best_model`.

## Start the MLflow tracking server

In the first terminal, with the environment activated in the project folder,
run:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

Keep the server running. In a second Git Bash terminal opened in the same
project folder, activate the environment and run the experiments:

```bash
source .venv/Scripts/activate
python train_mlflow.py
```

The script prints each model's accuracy and weighted F1 score, then reports
the best model and saves it to `models/best_model`. The tracking server must
be running at `http://127.0.0.1:5000` while the script logs its runs.

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in a browser on the same
computer to compare the runs in the `Handwritten-Digit-Classification`
experiment. Exact metric values may vary with package versions.

## Version the source with Git

Configure your Git identity once if needed:

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

Stage and commit the practicals you want to include:

```bash
git add README.md train_digits.py make_features.py retrieve_features.py feature_repo train_mlflow.py requirements.txt .gitignore
git commit -m "Add MLflow digit classification experiments"
git push
```

The virtual environment, generated models, MLflow database and artifacts, and
local Feast data/state are excluded from Git.
