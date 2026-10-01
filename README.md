# 🩺 Disease Prediction Using Machine Learning

An end-to-end machine learning web application that predicts the most likely disease based on user-selected symptoms.

The system uses an **ensemble of seven machine learning models** and combines their predicted probabilities to generate the final prediction. A Flask-based web interface allows users to select symptoms, submit them for analysis, and view the predicted disease, confidence level, probability distribution, and individual model predictions.

> **⚠️ Educational / Research Use Only**
>
> This application is developed for educational and research purposes. It is **not a medical diagnostic system** and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

---

## 🌐 Live Demo

**Live Application:**  
https://disease-prediction-xlb3.onrender.com

**Source Code:**  
https://github.com/aadarshmishra0709-web/disease-prediction

---

## 📌 Project Overview

Disease diagnosis can involve a large number of possible conditions with overlapping symptoms. This project demonstrates how machine learning classification algorithms can be used to estimate the most likely disease from a set of reported symptoms.

The system currently supports:

- **131 symptoms**
- **41 disease classes**
- **7 machine learning models**
- Ensemble prediction using **average class probabilities**
- Confidence estimation
- Top predicted conditions
- Individual model predictions
- Symptom search and selection
- Responsive web interface
- Flask REST API
- Cloud deployment using Render

---

## 🎯 Objectives

The main objectives of this project are:

1. Build a machine learning system for symptom-based disease prediction.
2. Compare predictions from multiple classification algorithms.
3. Combine model probabilities using an ensemble approach.
4. Provide an easy-to-use web interface.
5. Display prediction confidence and supporting probability information.
6. Deploy the application as a publicly accessible web service.

---

## 🧠 Machine Learning Approach

The system uses seven trained classification models:

| Model                     | Purpose                                  |
| ------------------------- | ---------------------------------------- |
| Logistic Regression       | Linear classification baseline           |
| Decision Tree             | Rule-based classification                |
| Random Forest             | Ensemble of decision trees               |
| Gradient Boosting         | Sequential boosting-based classification |
| K-Nearest Neighbors       | Distance-based classification            |
| Support Vector Classifier | Margin-based classification              |
| Naive Bayes               | Probabilistic classification             |

Instead of relying on a single model, the application obtains the class probabilities from every model and calculates their average.

### Ensemble Prediction

For each disease class:

```text
Average Probability =
(P₁ + P₂ + P₃ + P₄ + P₅ + P₆ + P₇) / 7
```

The final prediction is the disease with the highest average probability.

In simplified form:

```text
                    ┌── Logistic Regression ──┐
                    ├── Decision Tree ────────┤
                    ├── Random Forest ─────────┤
User Symptoms ──────┼── Gradient Boosting ─────┤
                    ├── KNN ──────────────────┤
                    ├── SVC ──────────────────┤
                    └── Naive Bayes ──────────┘
                              │
                              ▼
                    Model Probabilities
                              │
                              ▼
                     Average Probabilities
                              │
                              ▼
                    Highest Probability Class
                              │
                              ▼
                     Final Disease Estimate
```

This approach provides a combined prediction rather than depending on the output of only one classifier.

---

## 🔄 System Workflow

```text
User
 │
 ▼
Select Symptoms
 │
 ▼
Frontend Validation
 │
 ▼
POST /predict
 │
 ▼
Flask Backend
 │
 ▼
Input Validation
 │
 ▼
Feature Preparation
 │
 ▼
7 Machine Learning Models
 │
 ▼
Probability Predictions
 │
 ▼
Average Model Probabilities
 │
 ▼
Final Disease Prediction
 │
 ├── Predicted Disease
 ├── Confidence
 ├── Confidence Level
 ├── Top Conditions
 └── Individual Model Predictions
 │
 ▼
Web Interface
```

---

## 🏗️ Project Architecture

```text
                    ┌──────────────────────┐
                    │      Web Browser     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Flask - app.py   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     ml_core.py       │
                    └──────────┬───────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
       Preprocessor       Model Bundle      Metadata
                               │
             ┌─────────────────┼──────────────────┐
             │        │        │        │         │
             ▼        ▼        ▼        ▼         ▼
            LR       DT       RF       GB        KNN
             │        │        │        │         │
             └────────┴────────┴────────┴─────────┤
                                                   │
                                            SVC + Naive Bayes
                                                   │
                                                   ▼
                                        Average Probabilities
                                                   │
                                                   ▼
                                          Final Prediction
```

---

## 📊 Dataset

The project uses symptom-based disease classification data containing:

- Training data
- Testing data
- Symptom features
- Disease labels

The current trained system contains:

```text
131 Symptoms
41 Diseases
```

The dataset is represented in:

```text
dataset/
├── Training.csv
└── Testing.csv
```

---

## 🤖 Model Artifacts

All trained model artifacts are stored in the `models/` directory.

```text
models/
├── decision_tree.joblib
├── gradient_boosting.joblib
├── knn.joblib
├── label_encoder.joblib
├── logistic_regression.joblib
├── metadata.json
├── naive_bayes.joblib
├── preprocessor.joblib
├── random_forest.joblib
└── svc.joblib
```

The application loads these artifacts when the Flask server starts.

Example startup message:

```text
Loaded 7 models, 131 symptoms, 41 diseases.
```

---

## 🖥️ Web Application Features

### 1. Symptom Selection

Users can:

- Search through available symptoms
- Select multiple symptoms
- Remove selected symptoms
- View the number of selected symptoms

The application recommends selecting at least **3 symptoms** for a more informative prediction.

### 2. Patient Information

The interface optionally accepts:

- Age
- Gender
- Height
- Weight

These values are validated and displayed as contextual information.

**Important:** the current trained model uses symptoms only. These demographic fields do **not** influence the machine learning prediction.

### 3. Prediction Result

The application displays:

- Predicted disease
- Prediction confidence
- Confidence level
- Top probable conditions
- Individual model predictions
- Number of symptoms used
- Relevant warnings

### 4. Confidence Levels

The application categorizes confidence as:

```text
High     >= 70%
Medium   >= 40%
Low      < 40%
```

These thresholds are application-level indicators and should not be interpreted as medical certainty.

---

## 🔌 API Endpoints

### `GET /`

Loads the main disease prediction interface.

### `GET /health`

Returns the application health status.

Example:

```json
{
  "status": "ok",
  "error": null
}
```

### `POST /predict`

Accepts selected symptoms and returns the prediction.

Example request:

```json
{
  "symptoms": ["fever", "headache", "fatigue"],
  "patient": {
    "age": 25,
    "gender": "male"
  }
}
```

The backend primarily uses the `symptoms` field for prediction.

Example response structure:

```json
{
  "prediction": "Example Disease",
  "confidence": 0.82,
  "confidence_level": "high",
  "model_predictions": {},
  "class_probabilities": {},
  "symptoms_used": 3,
  "patient_info_used_by_model": false,
  "warnings": [],
  "disclaimer": "..."
}
```

---

## 📁 Project Structure

```text
disease-prediction/
│
├── app.py
├── ml_core.py
├── trained_model.py
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── Training.csv
│   └── Testing.csv
│
├── models/
│   ├── decision_tree.joblib
│   ├── gradient_boosting.joblib
│   ├── knn.joblib
│   ├── label_encoder.joblib
│   ├── logistic_regression.joblib
│   ├── metadata.json
│   ├── naive_bayes.joblib
│   ├── preprocessor.joblib
│   ├── random_forest.joblib
│   └── svc.joblib
│
├── reports/
│   ├── ensemble_confusion_matrix.csv
│   └── metrics.json
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
│
├── templates/
│   └── index.html
│
└── tests/
    └── test_app.py
```

---

## ⚙️ Technologies Used

### Backend

- Python
- Flask
- Gunicorn

### Machine Learning

- NumPy
- Pandas
- Scikit-learn
- Joblib

### Frontend

- HTML5
- CSS3
- JavaScript
- Jinja2

### Deployment

- Git
- GitHub
- Render

---

## 🚀 Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/aadarshmishra0709-web/disease-prediction.git
```

Navigate into the project:

```bash
cd disease-prediction
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

The application will be available at:

```text
http://127.0.0.1:5000
```

---

## 🌐 Deployment

The application is deployed using **Render**.

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
gunicorn app:app
```

### Production Architecture

```text
GitHub Repository
       │
       ▼
     Render
       │
       ▼
Python Environment
       │
       ▼
Install Dependencies
       │
       ▼
Gunicorn
       │
       ▼
Flask Application
       │
       ▼
Machine Learning Models
       │
       ▼
Public HTTPS Application
```

---

## 🧪 Testing

The project includes automated tests under:

```text
tests/test_app.py
```

Run the tests using:

```bash
pytest
```

If `pytest` is not installed:

```bash
pip install pytest
```

---

## 📈 Reports

Model evaluation outputs are stored in:

```text
reports/
├── ensemble_confusion_matrix.csv
└── metrics.json
```

These files can be used to analyze model performance and the ensemble prediction behavior.

---

## 🔐 Input Validation

The application validates:

### Age

```text
0 - 120 years
```

### Height

```text
30 - 260 cm
```

### Weight

```text
1 - 500 kg
```

### Gender

```text
Male
Female
Other
```

The application also:

- Removes duplicate symptoms
- Rejects unknown symptoms
- Requires at least one symptom
- Validates numeric values
- Handles missing model artifacts
- Handles prediction errors

---

## ⚠️ Limitations

This project has several important limitations.

### 1. Not a medical diagnostic system

The predictions are generated by machine learning models trained on a dataset. They should not be treated as professional medical diagnoses.

### 2. Dataset limitations

Model performance depends heavily on the quality, coverage, balance, and representativeness of the training dataset.

### 3. Symptoms-only prediction

The current model uses symptoms as its predictive features. It does not incorporate:

- Laboratory tests
- Medical imaging
- Clinical history
- Medication history
- Vital signs
- Doctor examination
- Electronic health records

### 4. Demographic information is not currently used

Age, gender, height, and weight are accepted by the interface but do not influence the current prediction model.

### 5. Confidence is not medical certainty

The reported confidence represents the model's probability output and ensemble behavior. It does not represent the probability that a patient actually has a disease.

---

## 🔮 Future Improvements

Potential future development includes:

- Larger and more diverse clinical datasets
- Additional machine learning algorithms
- Hyperparameter optimization
- Cross-validation
- Model calibration
- Explainable AI techniques
- SHAP-based feature explanations
- Integration of laboratory test results
- Integration of medical imaging
- Patient history features
- Better uncertainty estimation
- Improved model monitoring
- Authentication and secure user management
- Database integration
- More comprehensive automated testing

---

## 📚 Learning Outcomes

This project demonstrates practical implementation of:

- Data preprocessing
- Feature engineering
- Supervised machine learning
- Multi-model classification
- Ensemble learning
- Probability averaging
- Model serialization using Joblib
- Flask application development
- REST API development
- Frontend and backend integration
- Input validation
- Model deployment
- Git and GitHub workflow
- Cloud deployment using Render

---

## 👨‍💻 Author

**Aadarsh Mishra**

Computer Science & Engineering

### Project

**Disease Prediction Using Machine Learning**

---

## 📄 License

This project is intended for educational and research purposes.

Before using the project or its dataset for commercial or clinical applications, verify the applicable dataset, software, and data-use licenses.

---

## ⚠️ Disclaimer

This project is an educational and research demonstration of machine learning for symptom-based disease classification.

**It is not intended to diagnose, treat, cure, or prevent any disease.**

Always consult a qualified healthcare professional for medical advice, diagnosis, and treatment.
