# Customer Churn Prediction

An end-to-end machine learning project that predicts which telecom customers are likely to cancel, served through a REST API and a Streamlit web app.

**Business question:** which customers are likely to churn, so the company can step in and retain them?

## Results

Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, 21 columns, 26.5% churn rate). Evaluated on a 20% stratified test set (1,409 customers).

| Model | Churn recall | Churn precision | ROC AUC |
|---|---|---|---|
| Logistic Regression | 0.78 | 0.50 | 0.842 |
| Random Forest | 0.75 | 0.55 | 0.842 |
| **Logistic Regression, tuned threshold (0.576)** | **0.72** | **0.54** | 0.842 |

The final model is a logistic regression: it ties the random forest on ROC AUC, trains fast and is easy to explain. The decision threshold was chosen from cross-validated predictions on the training data only, so the test set stayed unseen.

## What drives churn

- **Contract type:** 42.7% of month-to-month customers churned, against 11.3% on one-year and 2.8% on two-year contracts.
- **Tenure:** customers who churned had a median tenure of 10 months, against 38 months for those who stayed.
- **Fibre optic internet:** 41.9% churn, against 19.0% for DSL customers.

## How it works

All preprocessing (scaling numeric features, one-hot encoding categorical ones) lives inside a single scikit-learn `Pipeline`. The saved model accepts raw customer data directly, so the API needs no separate encoders.

```
Streamlit form -> FastAPI validation -> Pipeline (scale + encode + model) -> churn probability -> Streamlit result
```

## Project structure

```
app/
  main.py            FastAPI app: /health and /predict, with input validation
  predictor.py       model loading and prediction logic
tests/
  test_api.py        5 automated tests
streamlit_app.py     Streamlit front end: a form that calls the API
churn_pipeline_v2.ipynb   EDA, modelling, threshold tuning
churn_pipeline.joblib     trained pipeline (scikit-learn 1.7.2)
churn_config.json         threshold, column order, valid category values
Telco-Customer-Churn.csv  dataset
```

## Run it locally

```bash
pip install -r requirements.txt
```

Open two terminals in the project folder.

**Terminal 1: the API**

```bash
uvicorn app.main:app --reload
```

API docs are at http://127.0.0.1:8000/docs (choose **POST /predict**, then **Try it out** and **Execute**).

**Terminal 2: the Streamlit app**

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501, fill in the customer form and click **Predict churn risk**. The app sends the details to the API and shows the result: a red box for high risk, a green one for low risk.

Example API response:

```json
{
  "churn_probability": 0.8913,
  "will_churn": true,
  "risk_level": "high",
  "threshold": 0.5757
}
```

Run the tests with `pytest`.

## Roadmap

- [x] Notebook with a single sklearn pipeline
- [x] FastAPI service with input validation and tests
- [x] Streamlit front end that calls the API
- [ ] Docker and docker-compose
- [ ] CI/CD with GitHub Actions
- [ ] Deployment on AWS
- [ ] Retraining and model tracking on Databricks (MLflow)

## Limitations

The dataset is a single snapshot of one telecom company, so the model hasn't been tested on other time periods or markets. Predictions show who is at risk, not why a specific customer will leave.
