from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

from common import (
    LandmarksConv1DModel,
    LandmarksWindowDataset,
    checkpoint_path,
    load_config,
    set_seed,
)


def feature_to_region(idx):
    if idx < 34:
        return f"Jaw ({idx})"
    elif idx < 42:
        return f"R.Eyebrow ({idx})"
    elif idx < 50:
        return f"L.Eyebrow ({idx})"
    elif idx < 62:
        return f"Nose ({idx})"
    elif idx < 74:
        return f"R.Eye ({idx})"
    elif idx < 86:
        return f"L.Eye ({idx})"
    elif idx < 116:
        return f"Outer Lip ({idx})"
    else:
        return f"Inner Lip ({idx})"


def plot_attributions(mean_attr, top10_idx, out_path="artifacts/captum_landmark_attributions.png"):
    labels = [feature_to_region(i) for i in top10_idx]
    values = [mean_attr[i] for i in top10_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(labels[::-1], values[::-1], color="steelblue")
    plt.xlabel("Mean Absolute Attribution")
    plt.title("Top 10 Landmark Features by Captum Attribution (Landmarks Modality)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")


def main():
    cfg = load_config("config.yaml")
    set_seed(42)

    device = torch.device("cpu")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path(cfg), map_location=device)

    # Load model
    model = LandmarksConv1DModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("Model loaded successfully.")

    # Load validation dataset
    val_ds = LandmarksWindowDataset(
        cfg,
        split="val",
        feature_mean=ckpt["feature_mean"],
        feature_std=ckpt["feature_std"],
        target_min=ckpt["target_min"],
        target_max=ckpt["target_max"],
    )
    print(f"Validation dataset loaded: {len(val_ds)} samples.")

    # Pick a small batch of samples to explain
    n_samples = 500
    x_batch = torch.stack([val_ds[i][0] for i in range(min(n_samples, len(val_ds)))])
    x_batch = x_batch.to(device).requires_grad_(True)

    # Run Integrated Gradients
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(x_batch)
    attributions, delta = ig.attribute(
        x_batch,
        baseline,
        return_convergence_delta=True,
    )

    print(f"Attributions shape: {attributions.shape}")
    print(f"Convergence delta (lower is better): {delta.mean().item():.6f}")

    # Average attributions across samples and time steps
    mean_attr = attributions.detach().cpu().numpy()
    mean_attr = np.abs(mean_attr).mean(axis=(0, 1))  # shape: (136,)

    # Top 10 most important features
    top10_idx = np.argsort(mean_attr)[::-1][:10]
    print("\nTop 10 most important landmark features:")
    for rank, idx in enumerate(top10_idx):
        print(f"  Rank {rank+1}: Feature {idx} | Attribution: {mean_attr[idx]:.6f}")

    # Save results
    out_path = "artifacts/captum_landmark_attributions.npz"
    np.savez(out_path, attributions=mean_attr, top10_idx=top10_idx)
    print(f"\nSaved attributions to {out_path}")

    # Plot
    plot_attributions(mean_attr, top10_idx)


if __name__ == "__main__":
    main()