import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# --- 1. Load and Clean Dataset ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "shelter_occupancy_prediction", "data", "processed", "merged_data.csv")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.upper()
df = df.replace([np.inf, -np.inf], np.nan)
df = df[df["STRAIN_SCORE"] < 100]
df = df.dropna(subset=["STRAIN_SCORE"])

# --- 2. Create Binary Label---
df["HIGH_STRAIN"] = (df["STRAIN_SCORE"] > 50).astype(int)

# --- 3. Feature Selection ---
features = ['SHELTER_CAPACITY', 'TOTAL_RAINFALL_IN', 'AVG_TEMP_F', 'AVG_UNEMPLOYMENT_RATE', 'EVICTION_COUNT']
target = "HIGH_STRAIN"
X = df[features]
y = df[target]

# --- 4. Impute and Scale ---
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# --- 5. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- 6. Train Classifier ---
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# --- 7. Evaluation Metrics ---
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Shelter Strain Classifier Results")
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)

# === 8. Plot Confusion Matrix ===
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Low", "High"], yticklabels=["Low", "High"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix: High vs Low Shelter Strain")
plt.tight_layout()
plt.show()

