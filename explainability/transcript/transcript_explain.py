"""
transcript_explain.py
=====================
Captum Integrated Gradients explainability for the Transcript LSTM model.

Produces:
  1. Top-N features by mean absolute attribution (horizontal bar chart)  ← matches reference style
  2. Per-feature attribution heatmap over time (mean across samples)
  3. Per-feature signed attribution bar chart (positive vs negative contributions)

Usage (from the src/ directory, with .venv activated):
    python transcript/transcript_explain.py --config transcript/config.yaml

Optional flags:
    --checkpoint   path/to/transcript_lstm.pt   (overrides config default)
    --output-dir   transcript/artifacts/explanations
    --top-n        10    (number of features shown in bar chart)
    --max-samples  200   (cap on windows used for attribution — keeps it fast)
    --subject      3     (restrict to a single subject; 0 = all subjects)
    --story        2     (restrict to a single story; 0 = all stories)
    --device       auto
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Feature names (11 total — must match preprocessing order)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "Warriner: Valence",
    "Warriner: Arousal",
    "Warriner: Dominance",
    "DepecheMood: Afraid",
    "DepecheMood: Amused",
    "DepecheMood: Angry",
    "DepecheMood: Annoyed",
    "DepecheMood: Dont_Care",
    "DepecheMood: Happy",
    "DepecheMood: Inspired",
    "DepecheMood: Sad",
]

# ---------------------------------------------------------------------------
# Captum wrapper — collapses (x, subject_idx) into a single float tensor
# so Integrated Gradients works on x only while subject_idx stays fixed
# ---------------------------------------------------------------------------
class TranscriptForCaptum(nn.Module):
    """Wrap the model so Captum sees exactly one input tensor (x_windows).

    subject_idx is injected at construction time and kept fixed during
    attribution — we attribute the input features, not the subject ID.
    """

    def __init__(self, model: nn.Module, subject_idx: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("subject_idx", subject_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sid = self.subject_idx.expand(x.shape[0])
        return self.model(x, sid)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_and_checkpoint(cfg: dict, ckpt_path: Path, device: torch.device):
    """Load TranscriptLSTMModel from a checkpoint."""
    try:
        from common import TranscriptLSTMModel
    except ImportError:
        from common import TranscriptLSTMModel

    saved = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TranscriptLSTMModel(cfg).to(device)
    model.load_state_dict(saved["model_state"])
    model.eval()
    return model, saved

def collect_windows(
    cfg: dict,
    split: str = "val",
    max_samples: int = 200,
    subject_filter: int = 0,
    story_filter: int = 0,
) -> list[tuple[np.ndarray, int]]:
    """Return a list of (window_array [100,11], subject_id_0indexed) pairs."""
    from common import iter_samples, read_features, window_features

    results: list[tuple[np.ndarray, int]] = []
    all_samples = iter_samples(cfg, split)
    print(f"  iter_samples returned {len(all_samples)} samples for split='{split}'")
    for sample in all_samples:
        print(f"  Trying: subject={sample.subject} story={sample.story}")
        if subject_filter and sample.subject != subject_filter:
            continue
        if story_filter and sample.story != story_filter:
            continue
        try:
            x = read_features(cfg, sample)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            continue
        print(f"  Loaded features shape: {x.shape}")
        # try:
        #     x = read_features(cfg, sample)
        # except FileNotFoundError as e:
        #     print(f"  NOT FOUND: {e}")
        #     continue
# def collect_windows(
#     cfg: dict,
#     split: str = "val",
#     max_samples: int = 200,
#     subject_filter: int = 0,
#     story_filter: int = 0,
# ) -> list[tuple[np.ndarray, int]]:
#     """Return a list of (window_array [100,11], subject_id_0indexed) pairs."""
#     try:
#         from common import iter_samples, read_features, window_features
#     except ImportError:
#         from common import iter_samples, read_features, window_features

#     results: list[tuple[np.ndarray, int]] = []
#     for sample in iter_samples(cfg, split):
#         if subject_filter and sample.subject != subject_filter:
#             continue
#         if story_filter and sample.story != story_filter:
#             continue
#         try:
#             x = read_features(cfg, sample)
#         except FileNotFoundError:
#             print(f"  NOT FOUND: {e}")
#             continue
        windows = window_features(
            x,
            window_size=int(cfg["model"]["window_size"]),
            stride=int(cfg["model"]["stride"]),
        )
        print(f"  Windows generated: {len(windows)}")
        for w in windows:
            results.append((w, sample.subject - 1))  # 0-indexed subject
            if len(results) >= max_samples:
                return results
    return results


# def compute_integrated_gradients(
#     wrapped_model: TranscriptForCaptum,
#     x_batch: torch.Tensor,
#     n_steps: int = 50,
# ) -> np.ndarray:
#     """Run Integrated Gradients and return attributions [B, T, F]."""
#     from captum.attr import IntegratedGradients

#     ig = IntegratedGradients(wrapped_model)
#     baseline = torch.zeros_like(x_batch)

#     attrs = ig.attribute(
#         x_batch,
#         baselines=baseline,
#         n_steps=n_steps,
#         return_convergence_delta=False,
#     )
#     return attrs.detach().cpu().numpy()  # [B, T, F]

def compute_integrated_gradients(
    wrapped_model: TranscriptForCaptum,
    x_batch: torch.Tensor,
    n_steps: int = 50,
) -> np.ndarray:
    """Run Integrated Gradients and return attributions [B, T, F]."""
    from captum.attr import IntegratedGradients

    # LSTM backward requires training mode for cuDNN
    wrapped_model.train()

    ig = IntegratedGradients(wrapped_model)
    baseline = torch.zeros_like(x_batch)

    attrs = ig.attribute(
        x_batch,
        baselines=baseline,
        n_steps=n_steps,
        return_convergence_delta=False,
    )

    # Switch back to eval after attribution
    wrapped_model.eval()

    return attrs.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_top_n_bar(
    mean_abs_attr: np.ndarray,   # [F]
    feature_names: list[str],
    top_n: int,
    out_path: Path,
    title: str = "Top Transcript Features by Captum Attribution (Transcript Modality)",
) -> None:
    """Horizontal bar chart matching the reference style."""
    ranked = np.argsort(mean_abs_attr)[::-1][:top_n]
    # Reverse so highest is at top (matplotlib barh lists bottom→top)
    ranked = ranked[::-1]

    labels = [feature_names[i] for i in ranked]
    values = mean_abs_attr[ranked]

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.55)))
    ax.barh(labels, values, color="#1F77B4")  # same steel-blue as reference
    ax.set_xlabel("Mean Absolute Attribution")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    # Minimal frame — matches the clean reference look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_signed_bar(
    mean_attr: np.ndarray,   # [F] — signed mean (not abs)
    feature_names: list[str],
    out_path: Path,
    title: str = "Signed Mean Attribution per Feature (Transcript Modality)",
) -> None:
    """Signed bar chart — green = positive contribution, red = negative."""
    order = np.argsort(mean_attr)  # ascending
    labels = [feature_names[i] for i in order]
    values = mean_attr[order]
    colors = ["#2A9D8F" if v >= 0 else "#E63946" for v in values]

    fig, ax = plt.subplots(figsize=(10, max(4, len(feature_names) * 0.55)))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean Signed Attribution")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_temporal_heatmap(
    mean_attr_time: np.ndarray,  # [T, F] — mean over samples
    feature_names: list[str],
    out_path: Path,
    title: str = "Mean Attribution Over Time per Feature (Transcript Modality)",
) -> None:
    """Heatmap: x=timestep, y=feature, colour=mean abs attribution."""
    T, F = mean_attr_time.shape
    abs_attr = np.abs(mean_attr_time)  # [T, F]

    fig, ax = plt.subplots(figsize=(12, max(4, F * 0.5)))
    im = ax.imshow(
        abs_attr.T,       # [F, T]
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
    )
    ax.set_yticks(range(F))
    ax.set_yticklabels(feature_names, fontsize=9)
    ax.set_xlabel("Timestep (within window)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Mean |Attribution|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Captum IG explainability for Transcript LSTM model.")
    p.add_argument("--config",       type=str,  default="transcript/config.yaml")
    p.add_argument("--checkpoint",   type=str,  default="")
    p.add_argument("--output-dir",   type=str,  default="transcript/artifacts/explanations")
    p.add_argument("--top-n",        type=int,  default=11)  # show all 11 by default
    p.add_argument("--max-samples",  type=int,  default=200)
    p.add_argument("--subject",      type=int,  default=0,   help="0 = all subjects")
    p.add_argument("--story",        type=int,  default=2,   help="0 = all stories; default=2 (val)")
    p.add_argument("--n-steps",      type=int,  default=50,  help="IG integration steps")
    p.add_argument("--batch-size",   type=int,  default=64)
    p.add_argument("--device",       type=str,  default="auto")
    return p.parse_args()


def main():
    args = parse_args()

    # ── imports ──────────────────────────────────────────────────────────────
    try:
        from common import load_config, choose_device, checkpoint_path
    except ImportError:
        from common import load_config, choose_device, checkpoint_path

    import captum  # noqa: F401
    # ── config + device ──────────────────────────────────────────────────────
    cfg = load_config(args.config)
    device = choose_device(args.device)
    print(f"Device: {device}")

    ckpt = Path(args.checkpoint) if args.checkpoint else checkpoint_path(cfg)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    print(f"Checkpoint: {ckpt}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load model ───────────────────────────────────────────────────────────
    model, saved = load_model_and_checkpoint(cfg, ckpt, device)
    print("Model loaded.")

    # ── collect validation windows ───────────────────────────────────────────
    print(f"Collecting up to {args.max_samples} validation windows …")
    windows = collect_windows(
        cfg,
        split="val", # Change this to ""train" if you want to use training data.
        max_samples=args.max_samples,
        subject_filter=args.subject,
        story_filter=args.story,
    )
    if not windows:
        raise RuntimeError(
            "No validation windows found. "
            "Check that feature .npy files exist and the config paths are correct."
        )
    print(f"  Collected {len(windows)} windows.")

    # ── run Integrated Gradients in batches ──────────────────────────────────
    all_attrs: list[np.ndarray] = []  # each [B, T, F]

    print(f"Running Integrated Gradients (n_steps={args.n_steps}) …")
    i = 0
    while i < len(windows):
        batch = windows[i : i + args.batch_size]
        x_np = np.stack([w[0] for w in batch], axis=0)  # [B, T, F]
        sid = batch[0][1]  # use first sample's subject for this batch

        x_tensor = torch.tensor(x_np, dtype=torch.float32, requires_grad=True).to(device)
        sid_tensor = torch.tensor(sid, dtype=torch.long).to(device)

        wrapped = TranscriptForCaptum(model, sid_tensor)
        attrs = compute_integrated_gradients(wrapped, x_tensor, n_steps=args.n_steps)
        all_attrs.append(attrs)

        i += args.batch_size
        print(f"  Processed {min(i, len(windows))}/{len(windows)} windows", end="\r")

    print()

    # ── aggregate ────────────────────────────────────────────────────────────
    all_attrs_np = np.concatenate(all_attrs, axis=0)  # [N, T, F]
    print(f"Attribution array shape: {all_attrs_np.shape}  (samples x timesteps x features)")

    # Mean absolute attribution per feature  →  [F]
    mean_abs_per_feature = np.mean(np.abs(all_attrs_np), axis=(0, 1))

    # Signed mean per feature  →  [F]
    mean_signed_per_feature = np.mean(all_attrs_np, axis=(0, 1))

    # Mean absolute attribution over time  →  [T, F]
    mean_abs_over_time = np.mean(np.abs(all_attrs_np), axis=0)

    # ── save raw attributions ─────────────────────────────────────────────────
    np.save(out_dir / "ig_attributions.npy", all_attrs_np)
    np.save(out_dir / "mean_abs_per_feature.npy", mean_abs_per_feature)
    print(f"  Saved raw attributions → {out_dir / 'ig_attributions.npy'}")

    # Print ranking to console
    ranked_idx = np.argsort(mean_abs_per_feature)[::-1]
    print("\nFeature ranking by mean |attribution|:")
    for rank, idx in enumerate(ranked_idx, 1):
        print(f"  {rank:2d}. {FEATURE_NAMES[idx]:<30s}  {mean_abs_per_feature[idx]:.4e}")

    # ── plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")

    # Plot 1 — Top-N horizontal bar (reference style)
    top_n = min(args.top_n, len(FEATURE_NAMES))
    plot_top_n_bar(
        mean_abs_per_feature,
        FEATURE_NAMES,
        top_n=top_n,
        out_path=out_dir / "captum_transcript_attributions.png",
        title=f"Top {top_n} Transcript Features by Captum Attribution (Transcript Modality)",
    )

    # Plot 2 — Signed attribution bar (all features)
    plot_signed_bar(
        mean_signed_per_feature,
        FEATURE_NAMES,
        out_path=out_dir / "captum_transcript_signed.png",
    )

    # Plot 3 — Temporal heatmap
    plot_temporal_heatmap(
        mean_abs_over_time,
        FEATURE_NAMES,
        out_path=out_dir / "captum_transcript_temporal_heatmap.png",
    )

    print(f"\nDone. All outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
