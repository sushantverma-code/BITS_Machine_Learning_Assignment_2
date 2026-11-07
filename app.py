import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="BITS ML Assignment 2", layout="wide")

st.title("BITS Pilani M.Tech AIML - Machine Learning Assignment 2")
st.markdown("### Credit Card Default Classification Dashboard")

@st.cache_resource
def load_artifacts():
    scaler = joblib.load('model/scaler.joblib')
    models = {
        "Logistic Regression": joblib.load('model/logistic_regression.joblib'),
        "Decision Tree": joblib.load('model/decision_tree.joblib'),
        "KNN": joblib.load('model/knn.joblib'),
        "Naive Bayes": joblib.load('model/naive_bayes.joblib'),
        "Support Vector Machine": joblib.load('model/support_vector_machine.joblib')
    }
    return scaler, models

scaler, models = load_artifacts()

st.sidebar.header("Navigation & Upload")
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.success("Test dataset loaded successfully!")
    
    st.subheader("Test Data Preview")
    st.dataframe(test_df.head())
    
    if 'target' not in test_df.columns:
        st.error("Uploaded CSV must contain the 'target' column.")
    else:
        X_test = test_df.drop(columns=['target'])
        y_test = test_df['target']
        
        selected_model_name = st.selectbox("Select Classification Model", list(models.keys()))
        model = models[selected_model_name]
        
        # Preprocessing alignment
        if selected_model_name in ["Logistic Regression", "KNN", "Support Vector Machine"]:
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
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            
        with col_2:
            st.subheader("Classification Report")
            report = classification_report(y_test, preds, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).transpose())
else:
    st.info("Please upload `test_data.csv` via the sidebar to begin evaluation.")