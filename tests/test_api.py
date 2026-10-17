"""Run from the project root:  pytest
Needs churn_pipeline.joblib and churn_config.json in the root (run train_model.py first)."""
from fastapi.testclient import TestClient

from app.main import app

SAMPLE = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 3,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.5, "TotalCharges": 286.5,
}


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok", "model_loaded": True}


def test_predict_valid_customer():
    with TestClient(app) as client:
        r = client.post("/predict", json=SAMPLE)
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["risk_level"] in ("high", "low")


def test_loyal_customer_scores_lower_than_new_month_to_month():
    loyal = {**SAMPLE, "tenure": 60, "Contract": "Two year", "TotalCharges": 5700.0,
             "InternetService": "DSL", "StreamingTV": "No", "StreamingMovies": "No",
             "MonthlyCharges": 60.0, "PaymentMethod": "Credit card (automatic)"}
    with TestClient(app) as client:
        p_new = client.post("/predict", json=SAMPLE).json()["churn_probability"]
        p_loyal = client.post("/predict", json=loyal).json()["churn_probability"]
        assert p_loyal < p_new


def test_invalid_category_rejected():
    with TestClient(app) as client:
        r = client.post("/predict", json={**SAMPLE, "Contract": "Weekly"})
        assert r.status_code == 422


def test_negative_tenure_rejected():
    with TestClient(app) as client:
        r = client.post("/predict", json={**SAMPLE, "tenure": -5})
        assert r.status_code == 422
