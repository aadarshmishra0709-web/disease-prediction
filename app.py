"""Flask backend for the Disease Prediction demo.

Run:  python app.py   then open http://127.0.0.1:5000
"""
from __future__ import annotations

import logging

import numpy as np
from flask import Flask, jsonify, render_template, request

import ml_core as core

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("disease_prediction")

app = Flask(__name__)

DISCLAIMER = ("This prediction is for educational/research purposes only and should not be "
              "considered a medical diagnosis. Please consult a qualified healthcare "
              "professional for medical advice.")
MIN_SYMPTOMS = 1          # hard minimum
RECOMMENDED_SYMPTOMS = 3  # every training example has >= 3 symptoms
HIGH_CONF, MEDIUM_CONF = 0.70, 0.40
TOP_N = 5                 # probabilities returned to the UI

# Demographic limits (context only - the dataset has no such columns).
LIMITS = {"age": (0, 120), "height": (30, 260), "weight": (1, 500)}  # years, cm, kg
GENDERS = {"male", "female", "other"}

# ---- load artifacts once at start-up --------------------------------------- #
BUNDLE = None
LOAD_ERROR = None
try:
    BUNDLE = core.ModelBundle()
    log.info("Loaded %d models, %d symptoms, %d diseases.",
             len(BUNDLE.models), len(BUNDLE.feature_names), len(BUNDLE.class_names))
except Exception as exc:  # noqa: BLE001 - surface any loading problem to the UI
    LOAD_ERROR = f"{type(exc).__name__}: {exc}"
    log.error("Could not load model artifacts: %s", LOAD_ERROR)


# ---- validation ------------------------------------------------------------ #
def parse_number(payload: dict, field: str, errors: list):
    raw = payload.get(field)
    if raw in (None, ""):
        return None  # optional
    try:
        value = float(raw)
    except (TypeError, ValueError):
        errors.append(f"{field.capitalize()} must be a number.")
        return None
    lo, hi = LIMITS[field]
    if not np.isfinite(value) or not lo <= value <= hi:
        errors.append(f"{field.capitalize()} must be between {lo} and {hi}.")
        return None
    return value


def validate_payload(payload):
    """Return (clean_input, errors)."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return None, ["Request body must be a JSON object."]

    symptoms = payload.get("symptoms")
    if not isinstance(symptoms, list) or not all(isinstance(s, str) for s in symptoms):
        errors.append("'symptoms' must be a list of symptom names.")
        symptoms = []
    symptoms = list(dict.fromkeys(symptoms))  # de-duplicate, keep order
    known = set(BUNDLE.feature_names)
    unknown = [s for s in symptoms if s not in known]
    if unknown:
        errors.append("Unrecognised symptom(s) submitted.")
    if len(symptoms) < MIN_SYMPTOMS:
        errors.append("Please select at least one symptom.")

    patient = {f: parse_number(payload, f, errors) for f in LIMITS}
    gender = payload.get("gender")
    if gender not in (None, ""):
        if str(gender).lower() not in GENDERS:
            errors.append("Gender must be male, female or other.")
        else:
            patient["gender"] = str(gender).lower()
    return ({"symptoms": symptoms, "patient": patient}, errors)


def confidence_level(conf: float) -> str:
    return "high" if conf >= HIGH_CONF else "medium" if conf >= MEDIUM_CONF else "low"


# ---- routes ---------------------------------------------------------------- #
@app.get("/")
def index():
    symptoms = []
    if BUNDLE:
        symptoms = sorted(
            ({"key": k, "label": BUNDLE.metadata["symptom_labels"][k]} for k in BUNDLE.feature_names),
            key=lambda s: s["label"])
    return render_template("index.html", symptoms=symptoms, load_error=LOAD_ERROR,
                           disclaimer=DISCLAIMER, recommended=RECOMMENDED_SYMPTOMS, limits=LIMITS)


@app.get("/health")
def health():
    return jsonify({"status": "ok" if BUNDLE else "error", "error": LOAD_ERROR}), (200 if BUNDLE else 503)


@app.post("/predict")
def predict():
    if BUNDLE is None:
        return jsonify({"error": "Model files could not be loaded. Run 'python trained_model.py' first."}), 503

    payload = request.get_json(silent=True)
    clean, errors = validate_payload(payload)
    if errors:
        return jsonify({"error": " ".join(errors), "errors": errors}), 400

    try:
        result = BUNDLE.predict(clean["symptoms"])
    except Exception:  # noqa: BLE001
        log.exception("Prediction failed")
        return jsonify({"error": "Prediction failed due to an internal error."}), 500

    names = BUNDLE.class_names
    disp = BUNDLE.metadata["class_display_names"]
    avg = result["average"]
    top = int(np.argmax(avg))          # final_prediction = argmax(average_probability)
    confidence = float(avg[top])

    order = np.argsort(avg)[::-1][:TOP_N]
    model_predictions = {}
    for key, proba in result["per_model"].items():
        i = int(np.argmax(proba))
        model_predictions[BUNDLE.metadata["model_display_names"][key]] = {
            "prediction": disp[names[i]], "confidence": round(float(proba[i]), 4)}

    warnings = []
    if len(clean["symptoms"]) < RECOMMENDED_SYMPTOMS:
        warnings.append(f"Only {len(clean['symptoms'])} symptom(s) selected; predictions are more "
                        f"reliable with at least {RECOMMENDED_SYMPTOMS}.")
    level = confidence_level(confidence)
    if level == "low":
        warnings.append("Confidence is low: the selected symptoms do not clearly point to one disease.")

    return jsonify({
        "prediction": disp[names[top]],
        "confidence": round(confidence, 4),
        "confidence_level": level,
        "model_predictions": model_predictions,
        "class_probabilities": {disp[names[i]]: round(float(avg[i]), 4) for i in order},
        "symptoms_used": len(clean["symptoms"]),
        "patient_info_used_by_model": False,
        "warnings": warnings,
        "disclaimer": DISCLAIMER,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
