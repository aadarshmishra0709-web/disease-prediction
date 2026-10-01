"""Shared ML logic used by BOTH training (trained_model.py) and inference (app.py).

Keeping preprocessing, model definitions and the ensemble rule in one module
guarantees training-time and prediction-time behaviour cannot drift apart.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
RANDOM_STATE = 42

# file stem -> display name (order matters: it is the order of the averaging)
MODEL_NAMES = {
    "knn": "KNN",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "svc": "SVC",
    "naive_bayes": "Naive Bayes",
    "logistic_regression": "Logistic Regression",
}

# Purely cosmetic fixes of misspelt labels in the source dataset (display only;
# the model is trained on the original labels).
DISPLAY_FIXES = {
    "Peptic ulcer diseae": "Peptic ulcer disease",
    "Dimorphic hemmorhoids(piles)": "Dimorphic hemorrhoids (piles)",
    "Osteoarthristis": "Osteoarthritis",
    "(vertigo) Paroymsal Positional Vertigo": "(Vertigo) Paroxysmal Positional Vertigo",
    "hepatitis A": "Hepatitis A",
}


# --------------------------------------------------------------------------- #
# Cleaning helpers
# --------------------------------------------------------------------------- #
def clean_label(text: str) -> str:
    """Strip and collapse whitespace in a disease label ('Diabetes ' -> 'Diabetes')."""
    return re.sub(r"\s+", " ", str(text)).strip()


def clean_feature_name(name: str) -> str:
    """Normalise a symptom column name: 'spotting_ urination' -> 'spotting_urination'."""
    name = re.sub(r"\s+", "_", str(name).strip())
    return re.sub(r"_+", "_", name)


def symptom_display_name(feature: str) -> str:
    text = feature.replace("_", " ").strip()
    return text[:1].upper() + text[1:]


def display_disease(label: str) -> str:
    return DISPLAY_FIXES.get(label, label)


# --------------------------------------------------------------------------- #
# Preprocessing (the single object that turns user input into model input)
# --------------------------------------------------------------------------- #
class SymptomVectorizer:
    """Converts a list of symptom names into the 0/1 vector the models were trained on.

    The dataset's symptoms are already binary indicators, so no scaling or
    one-hot encoding is required (and none is applied) - the vectorizer's job is
    to guarantee the exact column order and to reject unknown symptoms.
    """

    def __init__(self, feature_names):
        self.feature_names_ = list(feature_names)
        self._index = {n: i for i, n in enumerate(self.feature_names_)}

    def transform(self, symptoms) -> np.ndarray:
        unknown = [s for s in symptoms if s not in self._index]
        if unknown:
            raise ValueError(f"Unknown symptom(s): {', '.join(map(str, unknown[:5]))}")
        row = np.zeros((1, len(self.feature_names_)), dtype=np.float64)
        for s in set(symptoms):
            row[0, self._index[s]] = 1.0
        return row


# --------------------------------------------------------------------------- #
# Models and ensemble
# --------------------------------------------------------------------------- #
def build_models() -> dict:
    return {
        "knn": KNeighborsClassifier(n_neighbors=5),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "svc": SVC(probability=True, random_state=RANDOM_STATE),
        "naive_bayes": GaussianNB(),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    }


def per_model_probabilities(models: dict, X: np.ndarray, n_classes: int) -> dict:
    """Return {model_key: (n_samples, n_classes) probability matrix}.

    Verifies each model's class columns are aligned with 0..n_classes-1 so
    averaging vectors from different models is always column-consistent.
    """
    out = {}
    for key, model in models.items():
        if not np.array_equal(model.classes_, np.arange(n_classes)):
            raise RuntimeError(f"Model '{key}' was not trained on all {n_classes} classes.")
        out[key] = model.predict_proba(X)
    return out


def ensemble_probabilities(per_model: dict) -> np.ndarray:
    """Equal-weight soft vote: the plain mean of the 7 probability vectors."""
    return np.mean(np.stack(list(per_model.values()), axis=0), axis=0)


# --------------------------------------------------------------------------- #
# Artifact loading
# --------------------------------------------------------------------------- #
class ModelBundle:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.vectorizer = joblib.load(models_dir / "preprocessor.joblib")
        self.label_encoder = joblib.load(models_dir / "label_encoder.joblib")
        self.models = {k: joblib.load(models_dir / f"{k}.joblib") for k in MODEL_NAMES}
        with open(models_dir / "metadata.json", encoding="utf-8") as fh:
            self.metadata = json.load(fh)
        self.class_names = list(self.label_encoder.classes_)

    @property
    def feature_names(self):
        return self.vectorizer.feature_names_

    def predict(self, symptoms) -> dict:
        X = self.vectorizer.transform(symptoms)
        per_model = per_model_probabilities(self.models, X, len(self.class_names))
        avg = ensemble_probabilities(per_model)[0]
        return {"average": avg, "per_model": {k: v[0] for k, v in per_model.items()}}
