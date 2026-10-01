# Disease Prediction (Educational Demo)

A Flask web app that predicts a likely disease from selected symptoms using an
**equal-weight soft-voting ensemble of 7 scikit-learn classifiers**.

> **Disclaimer:** This prediction is for educational/research purposes only and should not be
> considered a medical diagnosis. Please consult a qualified healthcare professional for medical advice.

## Features
- **Frontend (HTML/CSS/JS):** responsive form, searchable symptom checklist (131 symptoms) with removable chips, loading state, error messages, probability bars, per-model votes, low-confidence warnings, Reset button.
- **Backend (Flask):** `POST /predict` with input validation, graceful model-loading and prediction error handling, `GET /health`.
- **ML:** KNN, Decision Tree, Random Forest, Gradient Boosting, SVC(`probability=True`), Gaussian Naive Bayes, Logistic Regression.

## Dataset (`dataset/`)
`Training.csv` (4,920 rows) and `Testing.csv` (42 rows). Findings from analysing the actual files:

| Item | Finding |
|---|---|
| Target | `prognosis` - 41 diseases (trailing/double spaces in 3 labels are cleaned) |
| Features | 132 columns, **all binary 0/1 symptom indicators**. No numerical or other categorical features. |
| Missing values | None (the file has one empty trailing column, which is dropped) |
| Duplicate column name | `fluid_overload` appears twice: one all-zero, one informative. The informative one is kept. Final feature count: **131** |
| Column names | A few contain stray spaces/underscores (`spotting_ urination`); normalised to `spotting_urination` |
| **Duplicates** | **4,616 of 4,920 rows are exact duplicates.** Only **304 unique** rows (5-10 per disease; each disease = 120 copies of its patterns) |
| Class balance | Perfectly balanced raw (120 each); mildly imbalanced after de-dup (5-10 each) |
| Conflicts | No identical symptom pattern is labelled with two different diseases |
| `Testing.csv` | 41 of its 42 rows are identical to training rows, so it is **not** an independent test set and is not used |

**Why de-duplicate first?** Splitting the raw file randomly would place identical rows in both train and test and produce
meaningless ~100% scores. Training therefore de-duplicates, then makes a **stratified 80/20 split** (`random_state=42`).
No SMOTE/oversampling is used: after de-duplication the classes are only mildly imbalanced (5-10 rows), stratification keeps
each disease in both splits, and synthesising binary symptom vectors would create unrealistic patients.

**No age/height/weight/gender in the dataset.** The UI collects them as requested, but they cannot influence the model.
They are validated, shown only as context, and clearly labelled as unused in the UI and API (`patient_info_used_by_model: false`).

## ML architecture
- **Preprocessing:** symptoms are already 0/1, so no scaler or one-hot encoder is needed. A saved `SymptomVectorizer`
  (`models/preprocessor.joblib`) maps symptom names to the exact column order used in training and rejects unknown symptoms.
  The same class is used by training and by the API; the frontend only sends symptom names.
- **Ensemble:** each model returns a probability vector over the 41 diseases; the 7 vectors are averaged with equal weights;
  `final_prediction = argmax(average_probability)`. Class-column alignment is asserted for every model.
- **Metrics:** accuracy plus macro (primary) and weighted precision/recall/F1 on the test split, 5-fold stratified CV, and an
  ensemble confusion matrix (`reports/ensemble_confusion_matrix.csv`).
- **Interpreting the scores:** near-perfect results mostly reflect that this dataset is small and idealised (few distinct patterns per disease), not clinical accuracy. Decision Tree and Gradient Boosting score noticeably lower.

## Installation (Windows)
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
macOS/Linux: `source venv/bin/activate` instead of the activate line.

## Training
Trained models are already included. To retrain (also required if your scikit-learn version differs from the one that saved the `.joblib` files):
```bash
python trained_model.py
```

## Run the app
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

Optional tests: `python -m unittest discover -s tests -v`

## Project structure
```
disease_prediction/
├── app.py                 Flask app: validation, /predict, /health
├── ml_core.py             Shared preprocessing, model definitions, ensemble rule, artifact loader
├── trained_model.py       Load -> clean -> dedupe -> split -> train 7 models -> evaluate -> save
├── requirements.txt
├── dataset/               Training.csv, Testing.csv
├── models/                7 model .joblib files, preprocessor.joblib, label_encoder.joblib, metadata.json
├── reports/               metrics.json, ensemble_confusion_matrix.csv
├── templates/index.html
├── static/css/style.css
├── static/js/script.js
└── tests/test_app.py
```

## Prediction workflow
```
User Input -> Frontend -> Flask API -> Input Validation -> Preprocessing (SymptomVectorizer)
-> 7 ML Models -> Probability Predictions -> Average Probabilities -> argmax -> Final Disease -> Frontend Result
```

## API
`POST /predict` with `{"symptoms": ["itching","skin_rash"], "age": 30, "height": 170, "weight": 65, "gender": "male"}`
(only `symptoms` is required) returns `prediction`, `confidence`, `confidence_level`, `model_predictions`,
`class_probabilities` (top 5), `warnings`, `disclaimer`.

## Notes
- Confidence levels: high >= 70%, medium 40-70%, low < 40% (a warning is shown when low or when fewer than 3 symptoms are chosen).
- Dataset origin/licence: this appears to be a widely circulated public symptom-disease dataset; check its original licence before redistributing.
