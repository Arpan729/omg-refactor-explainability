from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from captum.attr import IntegratedGradients


class LinearStacker(nn.Module):
    def __init__(self, weights: np.ndarray, bias: float):
        super().__init__()
        self.linear = nn.Linear(len(weights), 1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor(weights, dtype=torch.float32).view(1, -1))
            self.linear.bias.copy_(torch.tensor([bias], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain the ensemble-fusion stacker with Captum.")
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_explain_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Explainability config must be a YAML dictionary.")

    base_dir = path.parent
    cfg.setdefault("captum", {})
    cfg["paths"] = {
        "ensemble_config": str(resolve_path(base_dir, cfg["paths"]["ensemble_config"])),
        "checkpoint": str(resolve_path(base_dir, cfg["paths"]["checkpoint"])),
        "output_dir": str(resolve_path(base_dir, cfg["paths"]["output_dir"])),
    }
    return cfg


def ensure_src_import_path(config_path: Path) -> None:
    src_dir = config_path.parent.parent
    if not (src_dir / "ensemble_fusion").exists():
        repo_src_dir = Path(__file__).resolve().parents[2] / "src"
        if (repo_src_dir / "ensemble_fusion").exists():
            src_dir = repo_src_dir
    sys.path.insert(0, str(src_dir))


def load_checkpoint(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        checkpoint = pickle.load(f)
    required = ["feature_names", "weights", "bias"]
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {missing}")
    return checkpoint


def build_validation_features(ensemble_cfg: dict[str, Any]):
    from ensemble_fusion.common import build_feature_frame, iter_samples

    xs: list[np.ndarray] = []
    subject_ids: list[np.ndarray] = []
    story_ids: list[np.ndarray] = []
    frame_indices: list[np.ndarray] = []
    feature_names: list[str] | None = None

    for sample in iter_samples(ensemble_cfg, "val"):
        x_sample, _, names = build_feature_frame(ensemble_cfg, sample, source="prediction")
        if feature_names is None:
            feature_names = list(names)
        elif feature_names != list(names):
            raise ValueError("Feature names differ across validation samples.")

        xs.append(x_sample.astype(np.float32))
        subject_ids.append(np.full(len(x_sample), sample.subject, dtype=np.int16))
        story_ids.append(np.full(len(x_sample), sample.story, dtype=np.int16))
        frame_indices.append(np.arange(len(x_sample), dtype=np.int32))

    if not xs or feature_names is None:
        raise FileNotFoundError("No validation samples found for ensemble explainability.")

    return (
        np.concatenate(xs, axis=0),
        np.concatenate(subject_ids, axis=0),
        np.concatenate(story_ids, axis=0),
        np.concatenate(frame_indices, axis=0),
        feature_names,
    )


def validate_feature_order(feature_names: list[str], checkpoint_feature_names: list[str]) -> None:
    if list(feature_names) != list(checkpoint_feature_names):
        raise ValueError(
            "Feature order mismatch between config and checkpoint: "
            f"built={feature_names}, checkpoint={checkpoint_feature_names}"
        )


def global_importance_frame(
    feature_names: list[str], attributions: np.ndarray, weights: np.ndarray
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "modality": feature_names,
            "mean_abs_attribution": np.mean(np.abs(attributions), axis=0),
            "signed_mean_attribution": np.mean(attributions, axis=0),
            "checkpoint_weight": weights,
        }
    )
    return df.sort_values("mean_abs_attribution", ascending=False).reset_index(drop=True)


def subject_importance_frame(
    feature_names: list[str], subject_ids: np.ndarray, attributions: np.ndarray
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for subject_id in sorted(np.unique(subject_ids).tolist()):
        mask = subject_ids == subject_id
        subject_attr = attributions[mask]
        for idx, name in enumerate(feature_names):
            rows.append(
                {
                    "subject_id": int(subject_id),
                    "modality": name,
                    "mean_abs_attribution": float(np.mean(np.abs(subject_attr[:, idx]))),
                    "signed_mean_attribution": float(np.mean(subject_attr[:, idx])),
                    "n_frames": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def plot_global_importance(df: pd.DataFrame, out_path: Path) -> None:
    plot_df = df.sort_values("mean_abs_attribution", ascending=True)
    plt.figure(figsize=(8, 4.8))
    plt.barh(plot_df["modality"], plot_df["mean_abs_attribution"], color="steelblue")
    plt.xlabel("Mean Absolute Attribution")
    plt.title("Ensemble Fusion Captum Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_subject_importance(df: pd.DataFrame, feature_names: list[str], out_path: Path) -> None:
    heatmap = df.pivot(index="subject_id", columns="modality", values="mean_abs_attribution")
    heatmap = heatmap.reindex(columns=feature_names)

    plt.figure(figsize=(9, 5))
    image = plt.imshow(heatmap.to_numpy(dtype=np.float32), aspect="auto", cmap="Blues")
    plt.xticks(np.arange(len(feature_names)), feature_names, rotation=30, ha="right")
    plt.yticks(np.arange(len(heatmap.index)), [f"S{int(v)}" for v in heatmap.index])
    plt.xlabel("Modality")
    plt.ylabel("Subject")
    plt.title("Per-Subject Ensemble Fusion Captum Importance")
    plt.colorbar(image, label="Mean Absolute Attribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    explain_cfg = load_explain_config(args.config)
    ensemble_config_path = Path(explain_cfg["paths"]["ensemble_config"])
    checkpoint_path = Path(explain_cfg["paths"]["checkpoint"])
    output_dir = Path(explain_cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_src_import_path(ensemble_config_path)
    from ensemble_fusion.common import load_config

    ensemble_cfg = load_config(ensemble_config_path)
    checkpoint = load_checkpoint(checkpoint_path)

    x_np, subject_ids, story_ids, frame_indices, feature_names = build_validation_features(ensemble_cfg)
    validate_feature_order(feature_names, checkpoint["feature_names"])

    weights = np.asarray(checkpoint["weights"], dtype=np.float32).reshape(-1)
    bias = float(checkpoint["bias"])
    model = LinearStacker(weights=weights, bias=bias)
    model.eval()

    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    baseline = torch.zeros_like(x_tensor)
    ig = IntegratedGradients(model)
    internal_batch_size = int(explain_cfg["captum"].get("internal_batch_size", len(x_tensor)))
    internal_batch_size = max(internal_batch_size, len(x_tensor))
    attributions, delta = ig.attribute(
        x_tensor,
        baselines=baseline,
        target=0,
        n_steps=int(explain_cfg["captum"].get("n_steps", 64)),
        internal_batch_size=internal_batch_size,
        return_convergence_delta=True,
    )

    attr_np = attributions.detach().cpu().numpy().astype(np.float32)
    prediction_np = model(x_tensor).detach().cpu().numpy().reshape(-1).astype(np.float32)
    delta_np = delta.detach().cpu().numpy().astype(np.float32)

    global_df = global_importance_frame(feature_names, attr_np, weights)
    subject_df = subject_importance_frame(feature_names, subject_ids, attr_np)

    global_csv = output_dir / "captum_ensemble_global_importance.csv"
    subject_csv = output_dir / "captum_ensemble_subject_importance.csv"
    npz_path = output_dir / "captum_ensemble_attributions.npz"
    global_plot = output_dir / "captum_ensemble_global_importance.png"
    subject_plot = output_dir / "captum_ensemble_subject_importance.png"

    global_df.to_csv(global_csv, index=False)
    subject_df.to_csv(subject_csv, index=False)
    np.savez(
        npz_path,
        feature_names=np.asarray(feature_names),
        subject_ids=subject_ids,
        story_ids=story_ids,
        frame_indices=frame_indices,
        features=x_np.astype(np.float32),
        attributions=attr_np,
        predictions=prediction_np,
        convergence_delta=delta_np,
    )
    plot_global_importance(global_df, global_plot)
    plot_subject_importance(subject_df, feature_names, subject_plot)

    print("Saved ensemble Captum artifacts:")
    print(f"  {global_csv}")
    print(f"  {subject_csv}")
    print(f"  {npz_path}")
    print(f"  {global_plot}")
    print(f"  {subject_plot}")
    print(f"Mean convergence delta: {float(np.mean(np.abs(delta_np))):.8f}")
    print(global_df.to_string(index=False))


if __name__ == "__main__":
    main()
