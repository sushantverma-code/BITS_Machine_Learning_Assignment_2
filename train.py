import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# 1. Load dataset (adjust filename if yours differs slightly)
# Inside train.py
df = pd.read_csv('DataSet/UCI_Credit_Card.csv')

# Clean column names or handle ID column if present
if 'ID' in df.columns:
    df = df.drop(columns=['ID'])

# Identify target column (commonly named 'default.payment.next.month' or 'default payment next month')
target_col = [col for col in df.columns if 'default' in col.lower()][0]

X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save test data for Streamlit upload requirement
test_data = X_test.copy()
test_data['target'] = y_test
os.makedirs('model', exist_ok=True)
test_data.to_csv('test_data.csv', index=False)

# Scale features for models sensitive to feature magnitudes
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'model/scaler.joblib')

# 2. Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}

results = []

for name, model in models.items():
    # Fit and predict using scaled data for LR, KNN, SVM; unscaled works fine for DT, NB
    if name in ["Logistic Regression", "KNN", "Support Vector Machine"]:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
    # Save model
    joblib.dump(model, f"model/{name.lower().replace(' ', '_')}.joblib")
    
    # Compute metrics
    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "AUC": round(roc_auc_score(y_test, probs), 4),
        "Precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "Recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "F1": round(f1_score(y_test, preds, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_test, preds), 4)
    })

results_df = pd.DataFrame(results)
print(results_df)