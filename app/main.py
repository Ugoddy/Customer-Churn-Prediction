"""FastAPI service for the churn model.

Run from the project root:  uvicorn app.main:app --reload
Interactive docs:           http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from app.predictor import ChurnModel

YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNoInternet
    OnlineBackup: YesNoNoInternet
    DeviceProtection: YesNoNoInternet
    TechSupport: YesNoNoInternet
    StreamingTV: YesNoNoInternet
    StreamingMovies: YesNoNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
                    "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
                    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
                    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
                    "StreamingMovies": "Yes", "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 95.5, "TotalCharges": 286.5,
                }
            ]
        }
    }


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    risk_level: Literal["high", "low"]
    threshold: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = ChurnModel()   # loaded once at startup
    yield


app = FastAPI(title="Customer Churn API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_loaded": request.app.state.model is not None}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer, request: Request):
    return request.app.state.model.predict(customer.model_dump())
