"""Model loading and prediction logic (no web code, so it is easy to test)."""
import json
import os
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "churn_pipeline.joblib"))
CONFIG_PATH = Path(os.getenv("CONFIG_PATH", ROOT / "churn_config.json"))


class ChurnModel:
    def __init__(self, model_path: Path = MODEL_PATH, config_path: Path = CONFIG_PATH):
        self.pipeline = joblib.load(model_path)
        with open(config_path) as f:
            self.config = json.load(f)
        self.threshold = float(self.config["threshold"])
        self.feature_order = list(self.config["feature_order"])

    def predict(self, record: dict) -> dict:
        """record: one customer using the original Telco column names."""
        X = pd.DataFrame([record])[self.feature_order]
        prob = float(self.pipeline.predict_proba(X)[0, 1])
        will_churn = prob >= self.threshold
        return {
            "churn_probability": round(prob, 4),
            "will_churn": will_churn,
            "risk_level": "high" if will_churn else "low",
            "threshold": round(self.threshold, 4),
        }
