# Handwritten Digit Classification — MLOps Practicals 2 and 5

Practical 2 builds and versions a baseline handwritten-digit classifier using
the Scikit-learn Digits dataset. Practical 5 stores the same 1,797 examples,
their 64 pixel features, and digit labels in a local Feast feature store.
These are learning demonstrations, not OCR systems for arbitrary camera images.

## Requirements

- Python 3.11 or 3.12
- Git

## Set up and run

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

The script prints the dataset shape, test accuracy, and a classification report
for digits 0–9. It saves the trained model to
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

## Version the source with Git

Create a separate empty GitHub repository for this member's work. Configure
your Git identity once if needed:

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

In the project folder, stage and commit the source and dependency list:

```bash
git init
git add README.md train_digits.py make_features.py retrieve_features.py feature_repo requirements.txt .gitignore
git commit -m "Add Feast feature store for digits"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/digits-practical-5-member-4.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. The virtual environment,
generated model, and local Feast data/state are excluded from Git; the
practicals version source rather than locally generated artifacts.
