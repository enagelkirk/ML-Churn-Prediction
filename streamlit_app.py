
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load feature columns
feature_columns = joblib.load("model/feature_columns.pkl")

# Load models
models = {
    "Logistic Regression": joblib.load("model/logistic_model.pkl"),
    "Random Forest (Original)": joblib.load("model/rf_original.pkl"),
    "Random Forest (Tuned)": joblib.load("model/rf_tuned.pkl")
}

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("data/Telco_Churn.csv")
    df.dropna(inplace=True)
    X = df[feature_columns]
    y = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)
    return X, y

X, y = load_data()

# UI
st.title("Customer Churn Prediction App")
st.markdown("Use this app to test different machine learning models for predicting customer churn.")

# Sidebar
model_choice = st.sidebar.selectbox("Choose a Model", list(models.keys()))
model = models[model_choice]

# Prediction
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]
roc_score = roc_auc_score(y, y_prob)
report = classification_report(y, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y, y_pred)

# Results
st.subheader("Model Performance")
st.markdown(f"**ROC-AUC Score:** {roc_score:.2f}")
st.markdown("**Classification Report:**")
st.dataframe(pd.DataFrame(report).transpose())

st.markdown("**Confusion Matrix:**")
st.dataframe(pd.DataFrame(conf_matrix, columns=["Predicted No", "Predicted Yes"], index=["Actual No", "Actual Yes"]))
