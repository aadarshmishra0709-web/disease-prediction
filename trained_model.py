"""Training pipeline: load -> clean -> de-duplicate -> split -> train 7 models ->
evaluate (individual + equal-weight soft-voting ensemble) -> save artifacts.

Run:  python trained_model.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

import ml_core as core

TARGET_RAW = "prognosis"
TEST_SIZE = 0.20
CV_FOLDS = 5


# --------------------------------------------------------------------------- #
# 1. Load + clean
# --------------------------------------------------------------------------- #
def find_training_file() -> Path:
    for name in ("Training.csv", "training.csv"):
        p = core.DATASET_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Training.csv not found in {core.DATASET_DIR}")


def load_and_clean(path: Path):
    """Return (X_df, y_series, report dict). Prints a data-quality report."""
    with open(path, newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    # Header has an empty trailing column and a duplicated 'fluid_overload'.
    # Give columns positional names first so pandas does not silently mangle them.
    df = pd.read_csv(path, header=0, names=[f"c{i}" for i in range(len(header))])
    df.columns = header
    empty_cols = [i for i, h in enumerate(header) if not h.strip()]
    df = df.drop(df.columns[empty_cols], axis=1) if empty_cols else df

    report = {"raw_shape": list(df.shape)}
    cols = list(df.columns)
    dup_names = {c for c in cols if cols.count(c) > 1}
    # Duplicated column names: keep the occurrence that carries information.
    keep = []
    for i, c in enumerate(cols):
        if c in dup_names:
            same = [j for j, d in enumerate(cols) if d == c]
            best = max(same, key=lambda j: df.iloc[:, j].sum() if c != TARGET_RAW else 0)
            if i != best:
                continue
        keep.append(i)
    report["duplicate_column_names"] = sorted(dup_names)
    df = df.iloc[:, keep]

    df.columns = [c if c == TARGET_RAW else core.clean_feature_name(c) for c in df.columns]
    df[TARGET_RAW] = df[TARGET_RAW].map(core.clean_label)
    features = [c for c in df.columns if c != TARGET_RAW]

    report["missing_values"] = int(df.isna().sum().sum())
    report["non_binary_cells"] = int((~df[features].isin([0, 1])).sum().sum())
    zero_var = [c for c in features if df[c].nunique() < 2]
    report["zero_variance_features"] = zero_var
    report["duplicate_rows"] = int(df.duplicated().sum())

    df = df.dropna().drop(columns=zero_var)  # zero-variance columns carry no information
    features = [c for c in features if c not in zero_var]
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report["rows_before_dedup"], report["rows_after_dedup"] = before, len(df)

    # Same symptom pattern labelled with different diseases would be a real conflict.
    report["conflicting_symptom_patterns"] = int(
        (df.groupby(features)[TARGET_RAW].nunique() > 1).sum())
    report["n_features"] = len(features)
    report["n_classes"] = int(df[TARGET_RAW].nunique())
    report["class_counts_unique_rows"] = df[TARGET_RAW].value_counts().to_dict()
    return df[features].astype(float), df[TARGET_RAW], report


def print_data_report(r: dict) -> None:
    counts = pd.Series(r["class_counts_unique_rows"])
    print("=" * 52, "\nDATA QUALITY REPORT\n" + "=" * 52)
    print(f"Raw shape                  : {r['raw_shape']}")
    print(f"Duplicated column names    : {r['duplicate_column_names']}")
    print(f"Missing values             : {r['missing_values']}")
    print(f"Non-binary feature cells   : {r['non_binary_cells']}")
    print(f"Zero-variance features     : {r['zero_variance_features']} (dropped)")
    print(f"Duplicate rows             : {r['duplicate_rows']}")
    print(f"Rows before/after dedup    : {r['rows_before_dedup']} -> {r['rows_after_dedup']}")
    print(f"Conflicting symptom sets   : {r['conflicting_symptom_patterns']}")
    print(f"Features / classes         : {r['n_features']} / {r['n_classes']}")
    print(f"Unique rows per class      : min={counts.min()} max={counts.max()} mean={counts.mean():.1f}")
    print("NOTE: the raw file repeats each disease 120x, so a random split of the raw\n"
          "      rows would leak identical rows into the test set. Splitting is done\n"
          "      AFTER de-duplication, stratified by disease.\n")


# --------------------------------------------------------------------------- #
# 2. Metrics
# --------------------------------------------------------------------------- #
def metrics(y_true, y_pred) -> dict:
    out = {"accuracy": accuracy_score(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        out[f"precision_{avg}"], out[f"recall_{avg}"], out[f"f1_{avg}"] = p, r, f
    return out


def print_metrics(title: str, m: dict) -> None:
    print(title)
    print(f"  Accuracy          : {m['accuracy']:.4f}")
    print(f"  Precision (macro) : {m['precision_macro']:.4f}   (weighted: {m['precision_weighted']:.4f})")
    print(f"  Recall    (macro) : {m['recall_macro']:.4f}   (weighted: {m['recall_weighted']:.4f})")
    print(f"  F1 Score  (macro) : {m['f1_macro']:.4f}   (weighted: {m['f1_weighted']:.4f})\n")


def cross_validate(X, y, n_classes) -> dict:
    """Stratified k-fold on the de-duplicated rows. A 61-row test set is noisy,
    so CV gives a steadier estimate. Models are re-cloned for every fold."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=core.RANDOM_STATE)
    scores = {k: [] for k in list(core.MODEL_NAMES) + ["ensemble"]}
    for tr, te in skf.split(X, y):
        fold = {k: clone(m).fit(X[tr], y[tr]) for k, m in core.build_models().items()}
        pm = core.per_model_probabilities(fold, X[te], n_classes)
        for k, p in pm.items():
            scores[k].append(accuracy_score(y[te], p.argmax(1)))
        scores["ensemble"].append(accuracy_score(y[te], core.ensemble_probabilities(pm).argmax(1)))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in scores.items()}


# --------------------------------------------------------------------------- #
# 3. Main
# --------------------------------------------------------------------------- #
def main() -> int:
    core.MODELS_DIR.mkdir(exist_ok=True)
    core.REPORTS_DIR.mkdir(exist_ok=True)

    X_df, y_raw, data_report = load_and_clean(find_training_file())
    print_data_report(data_report)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    feature_names = list(X_df.columns)
    X = X_df.to_numpy()

    # Stratified split: every disease appears in both train and test.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=core.RANDOM_STATE)
    print(f"Train rows: {len(X_tr)}   Test rows: {len(X_te)}\n")

    models = {}
    for key, model in core.build_models().items():
        print(f"Training {core.MODEL_NAMES[key]} ...")
        models[key] = model.fit(X_tr, y_tr)

    per_model = core.per_model_probabilities(models, X_te, n_classes)
    all_metrics = {}
    print("\n" + "=" * 20, "MODEL PERFORMANCE", "=" * 20, "\n")
    for key, proba in per_model.items():
        all_metrics[key] = metrics(y_te, proba.argmax(1))
        print_metrics(core.MODEL_NAMES[key], all_metrics[key])
    print("=" * 60, "\n")

    ens_pred = core.ensemble_probabilities(per_model).argmax(1)
    all_metrics["ensemble"] = metrics(y_te, ens_pred)
    print_metrics("ENSEMBLE PERFORMANCE (equal-weight soft voting)", all_metrics["ensemble"])
    print("Averaging: 'macro' = unweighted mean over the 41 diseases (primary, because all\n"
          "classes matter equally); 'weighted' = weighted by class support.\n")

    print(f"Running {CV_FOLDS}-fold stratified cross-validation (accuracy, mean +/- std) ...")
    cv = cross_validate(X, y, n_classes)
    for k, (mu, sd) in cv.items():
        name = "Ensemble" if k == "ensemble" else core.MODEL_NAMES[k]
        print(f"  {name:<20}: {mu:.4f} +/- {sd:.4f}")

    # Confusion matrix of the ensemble (CSV - no plotting dependency needed).
    cm = confusion_matrix(y_te, ens_pred, labels=np.arange(n_classes))
    pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_csv(
        core.REPORTS_DIR / "ensemble_confusion_matrix.csv")

    # ---- save everything needed for inference ----
    for key, model in models.items():
        joblib.dump(model, core.MODELS_DIR / f"{key}.joblib")
    joblib.dump(core.SymptomVectorizer(feature_names), core.MODELS_DIR / "preprocessor.joblib")
    joblib.dump(le, core.MODELS_DIR / "label_encoder.joblib")
    metadata = {
        "feature_names": feature_names,
        "symptom_labels": {f: core.symptom_display_name(f) for f in feature_names},
        "class_names": list(le.classes_),
        "class_display_names": {c: core.display_disease(c) for c in le.classes_},
        "model_display_names": core.MODEL_NAMES,
        "ensemble": "equal-weight soft voting (mean of 7 probability vectors)",
        "random_state": core.RANDOM_STATE,
        "data_report": data_report,
    }
    with open(core.MODELS_DIR / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    with open(core.REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump({"test_split": all_metrics,
                   "cross_validation_accuracy": {k: {"mean": m, "std": s} for k, (m, s) in cv.items()}},
                  fh, indent=2)

    # ---- verify saved artifacts reload and reproduce the ensemble ----
    bundle = core.ModelBundle()
    re_pred = core.ensemble_probabilities(
        core.per_model_probabilities(bundle.models, X_te, n_classes)).argmax(1)
    assert np.array_equal(re_pred, ens_pred), "Reloaded models disagree with in-memory models!"
    print("\nArtifacts saved to ./models and ./reports and reload check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
