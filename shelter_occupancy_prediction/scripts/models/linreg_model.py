import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# --- 1. Load Dataset ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", ".."))
MERGED_FILE = os.path.join(PROJECT_ROOT, "shelter_occupancy_prediction", "data", "processed", "merged_data.csv")

df = pd.read_csv(MERGED_FILE)
df.columns = df.columns.str.strip().str.upper()

# --- 2. Clean STRAIN_SCORE column ---
df = df.replace([np.inf, -np.inf], np.nan)
df = df[df["STRAIN_SCORE"] < 100]  # Filter out extreme outliers
df = df.dropna(subset=["STRAIN_SCORE"])  # Drop rows with NaN strain

# Debug summary
print("STRAIN_SCORE distribution:")
print(df["STRAIN_SCORE"].describe())

# --- 3. Select Features and Target v
features = ['SHELTER_CAPACITY', 'TOTAL_RAINFALL_IN', 'AVG_TEMP_F', 'AVG_UNEMPLOYMENT_RATE', 'EVICTION_COUNT']
target = 'STRAIN_SCORE'

X = df[features]
y = df[target]

# Impute missing feature values
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# --- 4. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- 5. Train Model ---
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- 6. Evaluate ---
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Results:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# --- 7. Plot Actual vs Predicted ---
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Strain Score")
plt.ylabel("Predicted Strain Score")
plt.title("Model Performance")
plt.tight_layout()
plt.show()
