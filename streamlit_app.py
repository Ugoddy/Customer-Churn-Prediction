"""Streamlit front end for the churn API.

Start the API first:   uvicorn app.main:app --reload
Then in a 2nd window:  streamlit run streamlit_app.py
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")
st.title("📉 Customer Churn Predictor")
st.caption("Enter a customer's details. The app sends them to the FastAPI service, which returns a churn risk.")

YES_NO = ["Yes", "No"]
YES_NO_NOINT = ["Yes", "No", "No internet service"]

with st.form("customer"):
    st.subheader("Account")
    c1, c2 = st.columns(2)
    contract = c1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    tenure = c2.slider("Tenure (months)", 0, 72, 3)
    paperless = c1.selectbox("Paperless billing", YES_NO)
    payment = c2.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly = c1.number_input("Monthly charges", 0.0, 200.0, 95.5, step=0.5)
    total = c2.number_input("Total charges", 0.0, 20000.0, 286.5, step=10.0)

    st.subheader("Services")
    c3, c4 = st.columns(2)
    phone = c3.selectbox("Phone service", YES_NO)
    lines = c4.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
    internet = c3.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    security = c4.selectbox("Online security", YES_NO_NOINT, index=1)
    backup = c3.selectbox("Online backup", YES_NO_NOINT, index=1)
    protection = c4.selectbox("Device protection", YES_NO_NOINT, index=1)
    support = c3.selectbox("Tech support", YES_NO_NOINT, index=1)
    tv = c4.selectbox("Streaming TV", YES_NO_NOINT)
    movies = c3.selectbox("Streaming movies", YES_NO_NOINT)

    st.subheader("Customer")
    c5, c6 = st.columns(2)
    gender = c5.selectbox("Gender", ["Female", "Male"])
    senior = c6.selectbox("Senior citizen", ["No", "Yes"])
    partner = c5.selectbox("Partner", ["No", "Yes"])
    dependents = c6.selectbox("Dependents", ["No", "Yes"])

    submitted = st.form_submit_button("Predict churn risk", type="primary")

if submitted:
    payload = {
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone, "MultipleLines": lines,
        "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": protection, "TechSupport": support, "StreamingTV": tv,
        "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
    }
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        res = r.json()
    except requests.exceptions.ConnectionError:
        st.error("Can't reach the API. Start it first with: uvicorn app.main:app --reload")
    except requests.exceptions.RequestException as e:
        st.error(f"The API returned an error: {e}")
    else:
        prob = res["churn_probability"]
        st.divider()
        if res["will_churn"]:
            st.error(f"HIGH RISK: {prob:.0%} chance this customer churns")
        else:
            st.success(f"Low risk: {prob:.0%} chance this customer churns")
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(f"Flagged as high risk when the probability is at or above {res['threshold']:.0%}.")
        with st.expander("Raw API response"):
            st.json(res)
