import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("fusion_dataset.csv")

# Sample for stability/speed
n = min(5000, len(df))
df = df.sample(n=n, random_state=42)

features = ["speech", "landmarks", "transcript", "raw_face"]

X = df[features]
y = df["target"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Plot
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("shap_summary_3mod.png", bbox_inches="tight")

print("SHAP completed")
