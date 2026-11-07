import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
import streamlit as st

st.set_page_config(page_title="BITS ML Assignment 2", layout="wide")

st.title(
    "BITS Pilani M.Tech AIML - Machine Learning Assignment 2 (ID: 2026AJ05970)"
)
st.markdown("### Credit Card Default Classification Dashboard")


@st.cache_resource
def load_artifacts():
  scaler = joblib.load("model/scaler.joblib")
  models = {
      "Logistic Regression": joblib.load("model/logistic_regression.joblib"),
      "Decision Tree": joblib.load("model/decision_tree.joblib"),
      "KNN": joblib.load("model/knn.joblib"),
      "Naive Bayes": joblib.load("model/naive_bayes.joblib"),
      "Support Vector Machine": joblib.load(
          "model/support_vector_machine.joblib"
      ),
  }
  return scaler, models


scaler, models = load_artifacts()

st.sidebar.header("Navigation & Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload custom test_data.csv (Optional)", type=["csv"]
)

default_path = "test_data.csv"
test_df = None

if uploaded_file is not None:
  test_df = pd.read_csv(uploaded_file)
  st.sidebar.success("Custom dataset loaded successfully!")
elif os.path.exists(default_path):
  test_df = pd.read_csv(default_path)
  st.sidebar.info("Default test_data.csv loaded automatically.")
else:
  st.warning(
      "Please upload `test_data.csv` via the sidebar or ensure it is present in"
      " the repository."
  )
  st.stop()

st.subheader("Test Data Preview")
st.dataframe(test_df.head())

if "target" not in test_df.columns:
  st.error("Dataset must contain the 'target' column.")
else:
  X_test = test_df.drop(columns=["target"])
  y_test = test_df["target"]

  selected_model_name = st.selectbox(
      "Select Classification Model", list(models.keys())
  )
  model = models[selected_model_name]

  # Preprocessing alignment
  if selected_model_name in [
      "Logistic Regression",
      "KNN",
      "Support Vector Machine",
  ]:
    X_test_processed = scaler.transform(X_test)
  else:
    X_test_processed = X_test

  preds = model.predict(X_test_processed)
  probs = model.predict_proba(X_test_processed)[:, 1]

  # Metrics calculation
  acc = accuracy_score(y_test, preds)
  auc = roc_auc_score(y_test, probs)
  prec = precision_score(y_test, preds, zero_division=0)
  recall = recall_score(y_test, preds, zero_division=0)
  f1 = f1_score(y_test, preds, zero_division=0)
  mcc = matthews_corrcoef(y_test, preds)

  st.markdown(f"### Results for: **{selected_model_name}**")
  c1, c2, c3, c4, c5, c6 = st.columns(6)
  c1.metric("Accuracy", f"{acc:.4f}")
  c2.metric("AUC", f"{auc:.4f}")
  c3.metric("Precision", f"{prec:.4f}")
  c4.metric("Recall", f"{recall:.4f}")
  c5.metric("F1 Score", f"{f1:.4f}")
  c6.metric("MCC Score", f"{mcc:.4f}")

  st.markdown("---")
  col_1, col_2 = st.columns(2)
  with col_1:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

  with col_2:
    st.subheader("Classification Report")
    report = classification_report(
        y_test, preds, output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose())