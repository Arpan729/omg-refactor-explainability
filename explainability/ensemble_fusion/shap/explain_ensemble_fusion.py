"""
SHAP Explainability for Ensemble Fusion Model
==============================================
Produces:
  1. Global feature importance bar chart (mean |SHAP|)
  2. SHAP beeswarm summary plot
  3. Per-subject SHAP bar charts (Story 2 val set)
  4. Per-frame SHAP time-series overlaid with ground-truth valence
  5. SHAP values CSV for downstream analysis

Usage (from the repo root, same working directory used for train/predict):
    python ensemble_fusion/explain_ensemble_fusion.py
    python ensemble_fusion/explain_ensemble_fusion.py --config ensemble_fusion/config.yaml
    python ensemble_fusion/explain_ensemble_fusion.py --output-dir ensemble_fusion/artifacts/shap_analysis

Dependencies:
    pip install shap matplotlib seaborn pandas numpy scikit-learn
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

try:
    from common import (
        SampleIndex,
        build_feature_frame,
        build_training_matrix,
        checkpoint_path,
        iter_samples,
        load_config,
    )
except ImportError:
    from ensemble_fusion.common import (
        SampleIndex,
        build_feature_frame,
        build_training_matrix,
        checkpoint_path,
        iter_samples,
        load_config,
    )

# ── Matplotlib style ────────────────────────────────────────────────────────
MODALITY_COLORS = {
    "speech":     "#4C78A8",
    "raw_face":   "#F58518",
    "transcript": "#54A24B",
    "landmarks":  "#B279A2",
    "fullbody":   "#9C755F",
}
plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":    150,
})


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SHAP analysis for the ensemble fusion ElasticNet stacker."
    )
    p.add_argument("--config",     type=str, default="ensemble_fusion/config.yaml")
    p.add_argument("--output-dir", type=str, default="ensemble_fusion/artifacts/shap_analysis")
    p.add_argument(
        "--n-background",
        type=int,
        default=2000,
        help="Number of background samples for LinearExplainer (subsample of training matrix).",
    )
    p.add_argument(
        "--max-timeseries-plots",
        type=int,
        default=5,
        help="How many per-subject time-series SHAP plots to save.",
    )
    return p.parse_args()


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def load_checkpoint(ckpt_path: Path) -> dict[str, Any]:
    with ckpt_path.open("rb") as f:
        saved = pickle.load(f)
    required = {"model", "feature_names", "weights", "bias"}
    missing = required - set(saved.keys())
    if missing:
        raise ValueError(f"Checkpoint missing keys: {missing}")
    return saved


# ── Data builders ─────────────────────────────────────────────────────────────
def build_background(
    cfg: dict[str, Any], n_background: int, rng: np.random.Generator
) -> tuple[np.ndarray, list[str]]:
    """
    Build the SHAP background dataset from the full training matrix (OOF predictions).
    Randomly subsample to `n_background` rows for efficiency.
    """
    x_train, _, feature_names = build_training_matrix(cfg)
    if len(x_train) > n_background:
        idx = rng.choice(len(x_train), size=n_background, replace=False)
        x_train = x_train[idx]
    print(f"  Background dataset: {x_train.shape[0]} frames × {x_train.shape[1]} modalities")
    return x_train.astype(np.float64), feature_names


def build_val_matrix(
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str], list[SampleIndex]]:
    """
    Build X_val, y_val for Story 2 (validation set).
    Returns X (n_frames, 5), y (n_frames,), feature_names, and per-sample index list
    so we can later split back per subject.
    """
    val_samples = iter_samples(cfg, "val")
    xs, ys, lengths, names_ref = [], [], [], None
    samples_out: list[SampleIndex] = []

    for sample in val_samples:
        x, y, names = build_feature_frame(cfg, sample, source="prediction")
        if names_ref is None:
            names_ref = names
        elif names_ref != names:
            raise ValueError("Feature name mismatch across val samples")
        xs.append(x)
        ys.append(y)
        lengths.append(len(y))
        samples_out.append(sample)

    if not xs or names_ref is None:
        raise FileNotFoundError(
            "No val samples found. Check that Story 2 prediction parquets exist "
            "at the paths specified in config.yaml."
        )

    x_val = np.concatenate(xs, axis=0).astype(np.float64)
    y_val = np.concatenate(ys, axis=0).astype(np.float64)
    print(f"  Val dataset: {x_val.shape[0]} frames × {x_val.shape[1]} modalities "
          f"across {len(samples_out)} subjects")
    return x_val, y_val, names_ref, samples_out, lengths  # type: ignore[return-value]


# ── SHAP computation ──────────────────────────────────────────────────────────
def compute_shap_values(
    model,
    x_background: np.ndarray,
    x_val: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    """
    Use shap.LinearExplainer — the correct explainer for sklearn linear models.
    Returns shap_values of shape (n_val_frames, n_features).

    Why LinearExplainer (not GradientExplainer / DeepExplainer)?
    - The ensemble fusion model is a scikit-learn ElasticNet (linear model).
    - LinearExplainer computes exact SHAP values analytically from model coefficients
      and background feature correlations. No approximation needed.
    - GradientExplainer / DeepExplainer are for neural networks only.
    """
    print(f"  Fitting LinearExplainer on background ({x_background.shape[0]} rows)...")
    explainer = shap.LinearExplainer(
        model,
        x_background,
        feature_perturbation="interventional",   # decorrelates features; robust default
    )
    print(f"  Computing SHAP values for {x_val.shape[0]} val frames...")
    shap_values = explainer.shap_values(x_val)          # (n_frames, n_features)
    shap_values = np.asarray(shap_values, dtype=np.float64)
    print(f"  SHAP values shape: {shap_values.shape}")

    # Sanity check: mean(SHAP) + expected_value ≈ mean(model predictions)
    # (approximate because interventional perturbation may not be perfectly additive
    #  for correlated features)
    expected_val = float(explainer.expected_value)
    approx_pred = shap_values.sum(axis=1) + expected_val
    mean_shap_pred = float(np.mean(approx_pred))
    print(f"  Expected value (baseline): {expected_val:.6f}")
    print(f"  Mean SHAP-reconstructed prediction: {mean_shap_pred:.6f}")

    return shap_values, expected_val


# ── Plotting helpers ──────────────────────────────────────────────────────────
def _modality_color_list(feature_names: list[str]) -> list[str]:
    fallback = "#888888"
    return [MODALITY_COLORS.get(n, fallback) for n in feature_names]


def plot_global_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
    out_path: Path,
    title: str = "Global SHAP Feature Importance (Mean |SHAP|)",
) -> None:
    """Horizontal bar chart of mean absolute SHAP values per modality."""
    mean_abs = np.abs(shap_values).mean(axis=0)          # (n_features,)
    order = np.argsort(mean_abs)                          # ascending for horizontal bar

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = _modality_color_list(feature_names)
    bar_colors = [colors[i] for i in order]
    bars = ax.barh(
        [feature_names[i] for i in order],
        mean_abs[order],
        color=bar_colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_beeswarm(
    shap_values: np.ndarray,
    x_val: np.ndarray,
    feature_names: list[str],
    out_path: Path,
) -> None:
    """
    SHAP beeswarm (summary) plot: shows spread and direction of each modality's
    SHAP values across all val frames, coloured by feature magnitude.
    """
    explanation = shap.Explanation(
        values=shap_values,
        data=x_val,
        feature_names=feature_names,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.beeswarm(explanation, show=False, max_display=len(feature_names))
    fig = plt.gcf()
    fig.suptitle("SHAP Beeswarm Plot — Ensemble Fusion (Story 2 Val)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_per_subject_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
    samples: list[SampleIndex],
    lengths: list[int],
    out_path: Path,
) -> None:
    """
    Grouped bar chart: mean |SHAP| per modality, one bar group per subject.
    Lets you see which modalities drive the model differently across subjects.
    """
    n_subjects = len(samples)
    n_features = len(feature_names)
    colors = _modality_color_list(feature_names)

    # Split shap_values back into per-subject chunks
    split_pts = np.cumsum(lengths)[:-1].tolist()
    per_subject_shap = np.split(shap_values, split_pts) if split_pts else [shap_values]

    # mean |SHAP| per subject × modality → (n_subjects, n_features)
    subj_importance = np.array([np.abs(s).mean(axis=0) for s in per_subject_shap])

    x = np.arange(n_subjects)
    width = 0.8 / n_features
    offsets = np.linspace(-(n_features - 1) / 2, (n_features - 1) / 2, n_features) * width

    fig, ax = plt.subplots(figsize=(max(8, n_subjects * 0.9), 5))
    for fi, (fname, color, offset) in enumerate(zip(feature_names, colors, offsets)):
        ax.bar(
            x + offset,
            subj_importance[:, fi],
            width=width * 0.9,
            label=fname,
            color=color,
            edgecolor="white",
            linewidth=0.4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s.subject}" for s in samples], fontsize=9)
    ax.set_xlabel("Subject", fontsize=10)
    ax.set_ylabel("Mean |SHAP Value|", fontsize=10)
    ax.set_title("Per-Subject SHAP Feature Importance (Story 2 Val)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_shap_timeseries(
    shap_slice: np.ndarray,          # (n_frames, n_features)
    y_true_slice: np.ndarray,        # (n_frames,)
    feature_names: list[str],
    expected_value: float,
    subject_id: int,
    story_id: int,
    out_path: Path,
) -> None:
    """
    Two-panel time-series plot for a single subject/story:
      Top panel:  Ground-truth valence
      Bottom panel: Stacked SHAP contributions over time (area plot)
    This shows *when* each modality drives the ensemble prediction.
    """
    n_frames = len(y_true_slice)
    frame_idx = np.arange(n_frames)
    colors = _modality_color_list(feature_names)

    fig, (ax_gt, ax_shap) = plt.subplots(
        2, 1, figsize=(13, 6), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.8]},
    )

    # ── Top: ground truth ──────────────────────────────────────────────────
    ax_gt.plot(frame_idx, y_true_slice, color="#1D3557", linewidth=1.0, label="Ground Truth")
    ax_gt.set_ylabel("Valence", fontsize=9)
    ax_gt.set_title(
        f"SHAP Time Series — Subject {subject_id}, Story {story_id}",
        fontsize=11, fontweight="bold",
    )
    ax_gt.grid(alpha=0.3)
    ax_gt.legend(loc="upper right", fontsize=8)

    # ── Bottom: stacked SHAP area plot ────────────────────────────────────
    # Separate positive and negative contributions
    shap_pos = np.clip(shap_slice, 0, None)   # (n_frames, n_features)
    shap_neg = np.clip(shap_slice, None, 0)

    # Stack positive contributions upward from baseline
    baseline = np.full(n_frames, expected_value)
    top = baseline.copy()
    for fi, (fname, color) in enumerate(zip(feature_names, colors)):
        ax_shap.fill_between(
            frame_idx,
            top,
            top + shap_pos[:, fi],
            label=fname,
            color=color,
            alpha=0.8,
            linewidth=0,
        )
        top = top + shap_pos[:, fi]

    # Stack negative contributions downward from baseline
    bot = baseline.copy()
    for fi, (fname, color) in enumerate(zip(feature_names, colors)):
        ax_shap.fill_between(
            frame_idx,
            bot,
            bot + shap_neg[:, fi],
            color=color,
            alpha=0.8,
            linewidth=0,
        )
        bot = bot + shap_neg[:, fi]

    ax_shap.axhline(expected_value, color="black", linewidth=0.8, linestyle="--",
                    alpha=0.5, label=f"Baseline ({expected_value:.3f})")
    ax_shap.set_xlabel("Frame", fontsize=9)
    ax_shap.set_ylabel("SHAP Contribution to Prediction", fontsize=9)
    ax_shap.grid(alpha=0.3)
    ax_shap.legend(loc="upper right", fontsize=8, framealpha=0.7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_mean_shap_heatmap(
    shap_values: np.ndarray,
    feature_names: list[str],
    samples: list[SampleIndex],
    lengths: list[int],
    out_path: Path,
) -> None:
    """
    Heatmap of mean SHAP (signed) per subject × modality.
    Signed values reveal direction: positive = modality pushes prediction UP.
    """
    split_pts = np.cumsum(lengths)[:-1].tolist()
    per_subject_shap = np.split(shap_values, split_pts) if split_pts else [shap_values]
    # mean signed SHAP per subject × modality (n_subjects, n_features)
    mean_signed = np.array([s.mean(axis=0) for s in per_subject_shap])

    fig, ax = plt.subplots(figsize=(7, max(4, len(samples) * 0.45)))
    im = ax.imshow(mean_signed, aspect="auto", cmap="RdBu_r", vmin=-np.abs(mean_signed).max(),
                   vmax=np.abs(mean_signed).max())
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(samples)))
    ax.set_yticklabels([f"S{s.subject}" for s in samples], fontsize=9)
    ax.set_title("Mean Signed SHAP per Subject × Modality (Story 2)", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Mean SHAP Value", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── CSV export ────────────────────────────────────────────────────────────────
def export_shap_csv(
    shap_values: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    samples: list[SampleIndex],
    lengths: list[int],
    out_path: Path,
) -> None:
    """
    Export a flat CSV with columns:
      subject_id, story_id, frame_idx, y_true,
      shap_<modality> × 5,
      feat_<modality> × 5   (the actual feature values for reference)
    """
    # Build subject / story / frame_idx columns
    subject_col, story_col, frame_col = [], [], []
    for sample, length in zip(samples, lengths):
        subject_col.extend([sample.subject] * length)
        story_col.extend([sample.story] * length)
        frame_col.extend(range(length))

    data: dict[str, Any] = {
        "subject_id": subject_col,
        "story_id":   story_col,
        "frame_idx":  frame_col,
        "y_true":     y_val,
    }
    for fi, fname in enumerate(feature_names):
        data[f"shap_{fname}"] = shap_values[:, fi]
        data[f"feat_{fname}"] = x_val[:, fi]

    df = pd.DataFrame(data)
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.name}  ({len(df)} rows)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.output_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # ── 1. Load checkpoint ──────────────────────────────────────────────────
    ckpt_path = checkpoint_path(cfg)
    print(f"\n[1/5] Loading checkpoint: {ckpt_path}")
    saved = load_checkpoint(ckpt_path)
    model        = saved["model"]
    feature_names: list[str] = list(saved["feature_names"])
    weights: list[float]     = list(saved["weights"])
    bias: float              = float(saved["bias"])

    print(f"  Model type   : {type(model).__name__}")
    print(f"  Features     : {feature_names}")
    print(f"  Coefficients : {dict(zip(feature_names, [round(w, 6) for w in weights]))}")
    print(f"  Intercept    : {bias:.6f}")

    # ── 2. Build datasets ────────────────────────────────────────────────────
    print("\n[2/5] Building background (train OOF) and val (Story 2) datasets...")
    x_background, bg_names = build_background(cfg, n_background=args.n_background, rng=rng)
    x_val, y_val, val_names, val_samples, val_lengths = build_val_matrix(cfg)

    if bg_names != val_names:
        raise ValueError(
            f"Feature name mismatch between training background ({bg_names}) "
            f"and val set ({val_names}). Re-check config modality order."
        )
    if val_names != feature_names:
        raise ValueError(
            f"Feature name mismatch between checkpoint ({feature_names}) "
            f"and rebuilt val matrix ({val_names})."
        )

    # ── 3. Compute SHAP values ───────────────────────────────────────────────
    print("\n[3/5] Computing SHAP values (LinearExplainer)...")
    shap_values, expected_value = compute_shap_values(model, x_background, x_val, feature_names)

    # ── 4. Plots ─────────────────────────────────────────────────────────────
    print("\n[4/5] Generating plots...")

    plot_global_importance(
        shap_values, feature_names,
        plots_dir / "shap_global_importance.png",
    )

    plot_beeswarm(
        shap_values, x_val, feature_names,
        plots_dir / "shap_beeswarm.png",
    )

    plot_per_subject_importance(
        shap_values, feature_names, val_samples, val_lengths,
        plots_dir / "shap_per_subject_importance.png",
    )

    plot_mean_shap_heatmap(
        shap_values, feature_names, val_samples, val_lengths,
        plots_dir / "shap_subject_modality_heatmap.png",
    )

    # Per-subject time-series plots (up to --max-timeseries-plots subjects)
    split_pts = np.cumsum(val_lengths)[:-1].tolist()
    shap_per_subject = np.split(shap_values, split_pts) if split_pts else [shap_values]
    y_per_subject    = np.split(y_val, split_pts)       if split_pts else [y_val]

    for i, (sample, shap_sl, y_sl) in enumerate(zip(val_samples, shap_per_subject, y_per_subject)):
        if i >= args.max_timeseries_plots:
            break
        plot_shap_timeseries(
            shap_sl, y_sl, feature_names, expected_value,
            sample.subject, sample.story,
            plots_dir / f"shap_timeseries_subject_{sample.subject}_story_{sample.story}.png",
        )

    # ── 5. Export CSV ────────────────────────────────────────────────────────
    print("\n[5/5] Exporting SHAP values to CSV...")
    export_shap_csv(
        shap_values, x_val, y_val, feature_names,
        val_samples, val_lengths,
        out_dir / "shap_values.csv",
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n── SHAP Summary ────────────────────────────────────────────────")
    mean_abs  = np.abs(shap_values).mean(axis=0)
    mean_sign = shap_values.mean(axis=0)
    std_shap  = shap_values.std(axis=0)
    summary_df = pd.DataFrame({
        "modality":        feature_names,
        "coef":            weights,
        "mean_abs_shap":   mean_abs,
        "mean_shap":       mean_sign,
        "std_shap":        std_shap,
        "pct_contribution": 100 * mean_abs / mean_abs.sum(),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    print(summary_df.to_string(index=False, float_format="%.6f"))

    summary_csv = out_dir / "shap_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n  Summary CSV saved: {summary_csv}")
    print(f"\nDone. All outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()