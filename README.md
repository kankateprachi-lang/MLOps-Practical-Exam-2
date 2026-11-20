# Handwritten Digit Classification — Practical 2

This practical builds and versions a baseline handwritten-digit classifier using
the Scikit-learn Digits dataset. The dataset contains 1,797 8×8 grayscale images
represented by 64 features each. It is intended for learning and is not an OCR
system for arbitrary camera images.

## Requirements

- Python 3.11 or 3.12
- Git

## Set up and run

From Git Bash, create and activate a virtual environment, then install the
project dependencies:

```bash
python --version
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train_digits.py
```

If `python` is not recognized on Windows, use `py --version` and
`py -3.11 -m venv .venv` (or `py -3.12 -m venv .venv`) when creating the
environment. After activation, `python` should refer to that environment.

The script prints the dataset shape, test accuracy, and a classification report
for digits 0–9. It saves the trained model to
`models/digit_classifier.joblib`.

## Version the source with Git

Create a separate empty GitHub repository named
`digits-practical-2-member-1` for this member's work. Configure your Git
identity once if needed:

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

In the project folder, stage and commit the source and dependency list:

```bash
git init
git add README.md train_digits.py requirements.txt .gitignore
git commit -m "Add handwritten digit classifier"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/digits-practical-2-member-1.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. The virtual environment and
generated model are excluded from Git; this practical versions the code rather
than the locally generated model artifact.
