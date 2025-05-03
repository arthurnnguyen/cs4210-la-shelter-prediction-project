import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# --- 1. Load Data ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", ".."))
MERGED_FILE = os.path.join(PROJECT_ROOT, "shelter_occupancy_prediction", "data", "processed", "merged_data.csv")
df = pd.read_csv(MERGED_FILE)
df.columns = df.columns.str.strip().str.upper()

# --- 2. Clean Target ---
df = df.replace([np.inf, -np.inf], np.nan)
df = df[df["STRAIN_SCORE"] < 100]
df = df.dropna(subset=["STRAIN_SCORE"])

# --- 3. Select Features & Target ---
features = ['SHELTER_CAPACITY', 'TOTAL_RAINFALL_IN', 'AVG_TEMP_F', 'AVG_UNEMPLOYMENT_RATE', 'EVICTION_COUNT']
target = 'STRAIN_SCORE'

X = df[features]
y = df[target]

# --- 4. Impute + Scale ---
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# --- 5. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- 6. Train Random Forest ---
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- 7. Evaluation ---
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Random Forest Model Results:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# --- 8. Feature Importance ---
importances = model.feature_importances_
feat_names = features
importance_df = (pd.DataFrame({'Feature': feat_names, 'Importance': importances}).
                 sort_values(by='Importance', ascending=False))

plt.figure(figsize=(7, 4))
sns.barplot(data=importance_df, x='Importance', y='Feature')
plt.title("Feature Importance - Predicting Shelter Strain")
plt.tight_layout()
plt.show()

# --- 9. Actual vs Predicted ---
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Strain Score")
plt.ylabel("Predicted Strain Score")
plt.title("Actual vs. Predicted - Random Forest")
plt.tight_layout()
plt.show()


# --- 10. Rank SPAs (Service Planning Area) by Predicted Strain ---
df["PREDICTED_STRAIN"] = model.predict(X_scaled)
spa_ranking = df.groupby("SPA")["PREDICTED_STRAIN"].mean().sort_values(ascending=False).reset_index()

print("\nTop SPAs by Average Predicted Shelter Strain:")
print(spa_ranking)

# Plot SPAs Average Predicted Shelter Strain
plt.figure(figsize=(8, 4))
sns.barplot(data=spa_ranking, x="SPA", y="PREDICTED_STRAIN", palette="viridis")
plt.title("Average Predicted Shelter Strain by SPA")
plt.xlabel("SPA")
plt.ylabel("Predicted Strain Score")
plt.tight_layout()
plt.show()
