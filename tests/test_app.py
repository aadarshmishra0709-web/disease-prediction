"""Run:  python -m unittest discover -s tests -v   (requires trained models)"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app as app_module  # noqa: E402
import ml_core as core  # noqa: E402


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app_module.app.test_client()
        cls.bundle = app_module.BUNDLE
        assert cls.bundle is not None, "Run python trained_model.py first"

    def test_all_models_support_probabilities(self):
        X = self.bundle.vectorizer.transform(["itching", "skin_rash"])
        for key, model in self.bundle.models.items():
            p = model.predict_proba(X)
            self.assertEqual(p.shape, (1, len(self.bundle.class_names)), key)
            self.assertAlmostEqual(float(p.sum()), 1.0, places=5, msg=key)

    def test_ensemble_is_equal_weight_mean_of_seven_vectors(self):
        syms = ["itching", "skin_rash", "nodal_skin_eruptions"]
        X = self.bundle.vectorizer.transform(syms)
        manual = sum(m.predict_proba(X)[0] for m in self.bundle.models.values()) / 7
        r = self.bundle.predict(syms)
        self.assertEqual(len(self.bundle.models), 7)
        np.testing.assert_allclose(r["average"], manual)
        self.assertAlmostEqual(float(r["average"].sum()), 1.0, places=5)

    def test_predict_endpoint(self):
        res = self.client.post("/predict", json={
            "symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"],
            "age": 30, "height": 170, "weight": 65, "gender": "male"})
        self.assertEqual(res.status_code, 200)
        d = res.get_json()
        for k in ("prediction", "confidence", "model_predictions", "class_probabilities"):
            self.assertIn(k, d)
        self.assertEqual(len(d["model_predictions"]), 7)
        self.assertEqual(d["prediction"], max(d["class_probabilities"], key=d["class_probabilities"].get))
        self.assertEqual(d["prediction"], "Fungal infection")

    def test_validation(self):
        bad = [({}, 400), ({"symptoms": []}, 400), ({"symptoms": ["not_a_symptom"]}, 400),
               ({"symptoms": "itching"}, 400),
               ({"symptoms": ["itching"], "age": 500}, 400),
               ({"symptoms": ["itching"], "height": "abc"}, 400),
               ({"symptoms": ["itching"], "gender": "x"}, 400)]
        for payload, code in bad:
            self.assertEqual(self.client.post("/predict", json=payload).status_code, code, payload)
        self.assertEqual(self.client.post("/predict", data="not json").status_code, 400)

    def test_page_and_health(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_every_training_pattern_roundtrips_through_the_vectorizer(self):
        self.assertEqual(len(self.bundle.feature_names), 131)
        self.assertEqual(len(self.bundle.class_names), 41)


if __name__ == "__main__":
    unittest.main()
