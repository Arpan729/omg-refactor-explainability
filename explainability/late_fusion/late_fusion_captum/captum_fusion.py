import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from captum.attr import IntegratedGradients

# =========================
# Load fusion dataset
# =========================
df = pd.read_csv("fusion_dataset.csv")

features = ["speech", "landmarks", "transcript", "raw_face"]
X = df[features].values.astype(np.float32)
y = df["target"].values.astype(np.float32)

X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y).unsqueeze(1)

# =========================
# Simple surrogate fusion model
# =========================
class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


model = FusionNet()

# =========================
# Train quickly
# =========================
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(100):
    optimizer.zero_grad()
    preds = model(X_tensor)
    loss = loss_fn(preds, y_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/100 - Loss: {loss.item():.6f}")

# =========================
# Captum Integrated Gradients
# =========================
ig = IntegratedGradients(model)

baseline = torch.zeros_like(X_tensor)

attributions, delta = ig.attribute(
    X_tensor,
    baselines=baseline,
    return_convergence_delta=True
)

attr = attributions.detach().numpy()
mean_importance = np.mean(np.abs(attr), axis=0)

print("\nCaptum fusion modality importance:")
for f, val in zip(features, mean_importance):
    print(f"{f}: {val:.6f}")

# =========================
# Save numeric results
# =========================
importance_df = pd.DataFrame({
    "modality": features,
    "importance": mean_importance
})

importance_df = importance_df.sort_values(by="importance", ascending=False)
importance_df.to_csv("captum_fusion_importance.csv", index=False)

print("\nSaved: captum_fusion_importance.csv")
print(importance_df)

# =========================
# Save bar plot
# =========================
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))
plt.bar(importance_df["modality"], importance_df["importance"])
plt.ylabel("Mean Absolute Attribution")
plt.title("Captum IG Modality Importance (Fusion Surrogate Model)")
plt.tight_layout()
plt.savefig("captum_fusion_importance.png", dpi=300)

print("Saved: captum_fusion_importance.png")
