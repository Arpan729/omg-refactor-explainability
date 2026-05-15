import matplotlib.pyplot as plt
import pandas as pd

data = pd.DataFrame({
    "Modality": ["Raw Face", "Landmarks", "Transcript", "Speech"],
    "Importance": [0.099587, 0.095775, 0.071265, 0.070877]
})

data = data.sort_values("Importance", ascending=True)

plt.figure(figsize=(9, 6))

bars = plt.barh(
    data["Modality"],
    data["Importance"],
    color="#1f4e79",          # single professional blue
    edgecolor="black",
    linewidth=0.6
)

for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 0.001,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.3f}",
        va="center",
        fontsize=10
    )

plt.xlabel("Mean Absolute Attribution (Captum IG)", fontsize=12)
plt.ylabel("Modality", fontsize=12)
plt.title(
    "Late Fusion Explainability (Captum IG)\nModality Contribution Comparison",
    fontsize=14,
    fontweight="bold"
)

plt.xlim(0, 0.11)
plt.grid(axis="x", linestyle="--", alpha=0.35)
plt.box(False)
plt.tight_layout()

plt.savefig("fusion_captum_final.png", dpi=300, bbox_inches="tight")
print("Saved: fusion_captum_final.png")
