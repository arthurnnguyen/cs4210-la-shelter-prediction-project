import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# --- 1. Load Data ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", ".."))
MERGED_FILE = os.path.join(PROJECT_ROOT, "shelter_occupancy_prediction", "data", "processed", "merged_data.csv")
df = pd.read_csv(MERGED_FILE)
df.columns = df.columns.str.strip().str.upper()

# --- 2. Clean Target ---
df = df.replace([np.inf, -np.inf], np.nan)
df = df[df["STRAIN_SCORE"] < 100]
df = df.dropna(subset=["STRAIN_SCORE"])

# --- 3. Feature Selection ---
features = ['SHELTER_CAPACITY', 'TOTAL_RAINFALL_IN', 'AVG_TEMP_F', 'AVG_UNEMPLOYMENT_RATE', 'EVICTION_COUNT']
target = 'STRAIN_SCORE'

X = df[features]
y = df[target]

# --- 4. Impute & Scale ---
X = SimpleImputer(strategy="mean").fit_transform(X)
X = StandardScaler().fit_transform(X)

# --- 5. Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 6. Neural Network Model ---
model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# === 7. Evaluation ===
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Neural Network Results:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# === 8. Plot Actual vs. Predicted ===
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Strain Score")
plt.ylabel("Predicted Strain Score")
plt.title("Neural Network Regression")
plt.tight_layout()
plt.show()
