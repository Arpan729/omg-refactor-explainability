from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

from common import (
    FullBodyResNet3DModel,
    FullBodyWindowDataset,
    checkpoint_path,
    load_config,
    set_seed,
)


def plot_attributions(mean_attr, out_path="artifacts/captum_fullbody_attributions.png"):
    # mean_attr shape: (seq_len, H, W) — average over samples
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: mean attribution per frame
    frame_attr = mean_attr.mean(axis=(1, 2))
    axes[0].bar(range(len(frame_attr)), frame_attr, color="steelblue")
    axes[0].set_xlabel("Frame in sequence")
    axes[0].set_ylabel("Mean Absolute Attribution")
    axes[0].set_title("Attribution per Frame (Fullbody Modality)")

    # Plot 2: spatial attribution heatmap averaged over frames
    spatial_attr = mean_attr.mean(axis=0)
    im = axes[1].imshow(spatial_attr, cmap="hot")
    axes[1].set_title("Spatial Attribution Heatmap (Fullbody Modality)")
    plt.colorbar(im, ax=axes[1])

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
    model = FullBodyResNet3DModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("Model loaded successfully.")

    # Load just one sample directly
    val_ds = FullBodyWindowDataset(
        cfg,
        split="val",
        target_min=ckpt["target_min"],
        target_max=ckpt["target_max"],
    )
    print(f"Validation dataset loaded: {len(val_ds)} samples.")

    # Pick just 1 sample
    n_samples = 1
    x_batch = val_ds[0][0].unsqueeze(0).to(device).requires_grad_(True)

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

    # Average attributions across samples and channel dim
    mean_attr = attributions.detach().cpu().numpy()
    mean_attr = np.abs(mean_attr).mean(axis=(0, 1))  # shape: (seq_len, H, W)

    # Save results
    out_path = "artifacts/captum_fullbody_attributions.npz"
    np.savez(out_path, attributions=mean_attr)
    print(f"\nSaved attributions to {out_path}")

    # Plot
    plot_attributions(mean_attr)


if __name__ == "__main__":
    main()
