import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU for SHAP compatibility

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load models and preprocessing tools
feature_columns = joblib.load("model/feature_columns.pkl")
scaler = joblib.load("model/scaler.pkl")

models = {
    "Logistic Regression": joblib.load("model/logistic_model.pkl"),
    "Random Forest (Original)": joblib.load("model/rf_original.pkl"),
    "Random Forest (Tuned)": joblib.load("model/rf_tuned.pkl")
}

@st.cache_data
def load_data():
    X_test = joblib.load("model/X_test.pkl")
    y_test = joblib.load("model/y_test.pkl")
    return X_test, y_test

X, y = load_data()

# --- Sidebar Help ---
st.sidebar.title("How to Use This App")
st.sidebar.info(
    "Choose a model\n\n"
    "Fill out the customer info\n\n"
    "Click **Predict Churn** to see results\n\n"
    "You’ll also get insights into why the model made its prediction"
)

# --- App Title and Intro ---
st.title("Customer Churn Prediction App")
st.markdown(
    """
    This app uses **machine learning** to predict whether a customer is likely to churn.
    
    ### Model Options:
    - **Logistic Regression**: A simple, interpretable model
    - **Random Forest**: A powerful, flexible ensemble model
    - **Tuned Random Forest**: A more optimized version for better performance

    ### What does the prediction mean?
    - **Yes** = The customer is likely to **cancel service**
    - **No** = The customer is likely to **stay**
    """
)

# --- Model Selection ---
model_choice = st.sidebar.selectbox("Choose a Model", list(models.keys()))
st.sidebar.markdown(f"**Debug:** Loaded `{model_choice}`")
model = models[model_choice]

st.header("Try a Custom Prediction")
st.markdown("Fill in the form below to simulate a customer's information:")

# --- User Inputs ---
gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has a Partner", ["No", "Yes"])
dependents = st.selectbox("Has Dependents", ["No", "Yes"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=150.0, value=70.0)
with st.expander("What is Total Charges?"):
    st.markdown(
        "Total Charges is the **cumulative amount** the customer has paid "
        "since joining. It's roughly equal to `Monthly Charges × Tenure`. "
        "It helps indicate how long and how much a customer has invested in the service."
    )

total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=2000.0)


# --- Prediction Logic ---
if st.button("Predict Churn"):
    input_dict = {
        'gender': gender,
        'SeniorCitizen': 1 if senior == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df)

    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[feature_columns]
    input_encoded[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.transform(
        input_encoded[["tenure", "MonthlyCharges", "TotalCharges"]]
    )
    input_encoded = input_encoded.astype(np.float64)

    # --- Prediction Output ---
    y_pred = model.predict(input_encoded)[0]
    y_prob = model.predict_proba(input_encoded)[0, 1]
    prediction = "Yes" if y_pred == 1 else "No"

    st.header("🧾 Prediction Result")
    if prediction == "Yes":
        st.warning(f"The customer is likely to churn.\n**Probability:** `{y_prob:.2f}`")
    else:
        st.success(f"The customer is likely to stay.\n**Probability:** `{y_prob:.2f}`")

    # --- Explanation Section ---
    if "Logistic" in model_choice:
        st.subheader("Why This Prediction?")
        try:
            explainer = shap.LinearExplainer(model, X)
            shap_values = explainer.shap_values(input_encoded)

            fig = plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=input_encoded.iloc[0]
                ),
                max_display=10,
                show=False
            )
            st.pyplot(fig)
        except Exception as e:
            st.error(f"SHAP explanation failed: {str(e)}")

    else:
        st.subheader("Top Features Influencing Random Forest Prediction")
        try:
            importances = model.feature_importances_
            top_features = pd.Series(importances, index=feature_columns).sort_values(ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(8, 5))
            top_features.plot(kind='barh', ax=ax, color="steelblue")
            ax.invert_yaxis()
            ax.set_xlabel("Feature Importance Score")
            ax.set_title("Top 10 Important Features")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Feature importance display failed: {str(e)}")
