import matplotlib.pyplot as plt
import pandas as pd

# Data summarised from your final Captum results
data = pd.DataFrame({
    "Frequency Bin": [4, 1, 3, 10, 8, 11, 51, 2, 13, 6],
    "Occurrences":   [10, 10, 10, 10, 9, 9, 9, 8, 8, 6]
})

# Sort so highest appears at top in barh
data = data.sort_values("Occurrences", ascending=True)

plt.figure(figsize=(9, 6))

bars = plt.barh(
    data["Frequency Bin"].astype(str),
    data["Occurrences"],
    color="#1f4e79",          # single professional blue
    edgecolor="black",
    linewidth=0.6
)

# Value labels
for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 0.08,
        bar.get_y() + bar.get_height() / 2,
        f"{int(width)}",
        va="center",
        fontsize=10
    )

plt.xlabel("Number of Validation Subjects in Top-10 Ranking", fontsize=12)
plt.ylabel("Frequency Bin", fontsize=12)
plt.title(
    "Speech Modality Explainability (Captum IG)\nMost Consistently Important Frequency Bins",
    fontsize=14,
    fontweight="bold"
)

plt.xlim(0, 11)
plt.grid(axis="x", linestyle="--", alpha=0.35)
plt.box(False)
plt.tight_layout()

plt.savefig("speech_captum_final.png", dpi=300, bbox_inches="tight")
print("Saved: speech_captum_final.png")
