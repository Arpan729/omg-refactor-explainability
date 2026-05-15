import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# Load predictions for all modalities
modalities = {
    'landmarks': 'src/landmarks/artifacts/predictions',
    'fullbody': 'src/fullbody/artifacts/predictions',
    'speech': 'src/speech/artifacts/predictions',
    'transcript': 'src/transcript/artifacts/predictions',
    'raw_face': 'src/raw_face/artifacts/predictions',
}

subjects = range(1, 11)
story = 2

all_data = []

for subject in subjects:
    filename = f"Subject_{subject}_Story_{story}.parquet"
    row = {'subject_id': subject}
    valid = True
    for modality, path in modalities.items():
        filepath = Path(path) / filename
        if filepath.exists():
            df = pd.read_parquet(filepath)
            row[modality] = df['y_pred'].mean()
        else:
            valid = False
            break
    if valid:
        all_data.append(row)

df_fusion = pd.DataFrame(all_data)
print(df_fusion)

X = df_fusion[list(modalities.keys())].values
feature_names = list(modalities.keys())

# SHAP KernelExplainer
background = X
explainer = shap.KernelExplainer(lambda x: x.mean(axis=1), background)
shap_values = explainer.shap_values(X)

# Plot
shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig('explainability/shap_fusion_results.png')
plt.show()

# Bar chart of mean absolute SHAP values - sorted
mean_shap = np.abs(shap_values).mean(axis=0)
feature_names_list = list(modalities.keys())

# Sort by value
sorted_idx = np.argsort(mean_shap)
sorted_features = [feature_names_list[i] for i in sorted_idx]
sorted_values = mean_shap[sorted_idx]

plt.figure(figsize=(8, 5))
plt.barh(sorted_features, sorted_values, color='salmon')
plt.xlabel('Mean Absolute SHAP Value')
plt.title('Modality Contributions to Hybrid Fusion (SHAP)')
plt.tight_layout()
plt.savefig('explainability/hybrid_fusion/shap_fusion_bar_chart.png')
plt.show()
print("Bar chart saved!")

# Per-subject SHAP breakdown
plt.figure(figsize=(10, 6))
x = np.arange(len(df_fusion))
width = 0.15

for i, modality in enumerate(feature_names):
    plt.bar(x + i * width, np.abs(shap_values[:, i]), width, label=modality)

plt.xlabel('Subject')
plt.ylabel('Absolute SHAP Value')
plt.title('Per-Subject Modality Contributions (SHAP)')
plt.xticks(x + width * 2, [f'S{int(s)}' for s in df_fusion['subject_id']])
plt.legend()
plt.tight_layout()
plt.savefig('explainability/hybrid_fusion/shap_per_subject.png')
plt.show()
print("Per-subject chart saved!")


# Waterfall plot for subject 1
shap.plots.waterfall(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X[0],
    feature_names=feature_names
), show=False)
plt.tight_layout()
plt.savefig('explainability/hybrid_fusion/shap_waterfall_subject1.png')
plt.show()
print("Waterfall plot saved!")
print("Done! Saved to explainability/shap_fusion_results.png")