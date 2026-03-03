from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ccc_numpy, load_config


PRED_RE = re.compile(r"Subject_(\d+)_Story_(\d+)\.parquet$")
REQUIRED_COLUMNS = ["frame_idx", "y_pred", "subject_id", "story_id"]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate landmarks prediction parquet files with CCC and plots.")
    p.add_argument("--config", type=str, default="landmarks/config.yaml")
    p.add_argument("--output-dir", type=str, default="landmarks/artifacts/model_evaluation")
    p.add_argument("--max-plots", type=int, default=10)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _parse_sample(path: Path) -> tuple[int, int] | None:
    m = PRED_RE.search(path.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _align(pred_df: pd.DataFrame, y_true_full: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    warnings = []
    missing = [c for c in REQUIRED_COLUMNS if c not in pred_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame_idx = pred_df["frame_idx"].to_numpy(dtype=np.int64)
    y_pred = pred_df["y_pred"].to_numpy(dtype=np.float64)

    if pd.Index(frame_idx).duplicated().any() or not pd.Index(frame_idx).is_monotonic_increasing:
        raise ValueError("Invalid frame_idx: duplicates or not monotonic increasing")

    keep = np.isfinite(y_pred) & (frame_idx >= 0) & (frame_idx < len(y_true_full))
    if np.any(~keep):
        warnings.append(f"dropped={int(np.sum(~keep))}")

    frame_idx = frame_idx[keep]
    y_pred = y_pred[keep]
    y_true = y_true_full[frame_idx]

    return frame_idx, y_true.astype(np.float64), y_pred.astype(np.float64), warnings


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {k: float("nan") for k in ["ccc", "mae", "rmse", "pred_mean", "true_mean", "pred_std", "true_std"]}
    err = y_pred - y_true
    return {
        "ccc": float(ccc_numpy(y_true, y_pred)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "pred_mean": float(np.mean(y_pred)),
        "true_mean": float(np.mean(y_true)),
        "pred_std": float(np.std(y_pred)),
        "true_std": float(np.std(y_true)),
    }


def _prepare_dir(out_dir: Path, overwrite: bool):
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise RuntimeError(f"Output directory is not empty: {out_dir}. Use --overwrite to replace files.")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)


def _plot_series(frame_idx, y_true, y_pred, out_path, subject_id, story_id):
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(frame_idx, y_true, label="y_true", linewidth=1.0, color="#1D3557")
    ax.plot(frame_idx, y_pred, label="y_pred", linewidth=1.0, color="#E63946", alpha=0.85)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Valence")
    ax.set_title(f"Landmarks Time Series: Subject {subject_id}, Story {story_id}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_ccc(df: pd.DataFrame, out_path: Path):
    if df.empty:
        return
    plot_df = df.sort_values("ccc", ascending=True)
    labels = [f"S{int(r.subject_id)}-T{int(r.story_id)}" for r in plot_df.itertuples()]
    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.35)))
    ax.barh(labels, plot_df["ccc"].to_numpy(dtype=float), color="#4C78A8")
    ax.set_xlabel("CCC")
    ax.set_ylabel("Sample")
    ax.set_title("Landmarks CCC by Sample")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_evaluation(cfg: dict[str, Any], out_dir: Path, max_plots: int, overwrite: bool):
    _prepare_dir(out_dir, overwrite)

    pred_dir = Path(cfg["paths"]["prediction_dir"])
    ann_dir = Path(cfg["paths"]["val_ann_dir"])
    prediction_files = sorted(pred_dir.glob("*.parquet"))

    rows = []
    full_true = []
    full_pred = []
    series = []
    skipped = 0

    for path in prediction_files:
        sample = _parse_sample(path)
        if sample is None:
            skipped += 1
            continue
        subject_id, story_id = sample
        ann_path = ann_dir / f"Subject_{subject_id}_Story_{story_id}.csv"
        if not ann_path.exists():
            skipped += 1
            continue

        pred_df = pd.read_parquet(path)
        y_true_full = pd.read_csv(ann_path).iloc[:, 0].to_numpy(dtype=np.float64)
        try:
            frame_idx, y_true, y_pred, warnings = _align(pred_df, y_true_full)
        except ValueError as exc:
            rows.append({"subject_id": subject_id, "story_id": story_id, "n_points": 0, "ccc": float("nan"), "warnings": str(exc)})
            skipped += 1
            continue

        row = {"subject_id": subject_id, "story_id": story_id, "n_points": int(len(y_true)), **_metrics(y_true, y_pred), "warnings": ";".join(warnings)}
        rows.append(row)
        if len(y_true):
            full_true.append(y_true)
            full_pred.append(y_pred)
            series.append((subject_id, story_id, frame_idx, y_true, y_pred))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics_per_sample.csv", index=False)

    overall_ccc = float("nan")
    if full_true:
        overall_ccc = float(ccc_numpy(np.concatenate(full_true), np.concatenate(full_pred)))

    summary = {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "num_prediction_files": len(prediction_files),
        "num_samples_evaluated": int(len(df) - skipped),
        "num_samples_skipped": int(skipped),
        "overall_ccc": overall_ccc,
        "mean_sample_ccc": float(df["ccc"].dropna().mean()) if "ccc" in df else float("nan"),
    }
    with (out_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _plot_ccc(df, out_dir / "plots" / "ccc_by_sample.png")
    for i, (subject_id, story_id, frame_idx, y_true, y_pred) in enumerate(series[:max_plots], start=1):
        _plot_series(frame_idx, y_true, y_pred, out_dir / "plots" / f"series_{i:02d}_S{subject_id}_T{story_id}.png", subject_id, story_id)

    return df, summary


def main():
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()

    _, summary = run_evaluation(cfg, out_dir, args.max_plots, args.overwrite)
    print(f"Evaluation complete. overall_ccc={summary['overall_ccc']:.6f} output_dir={out_dir}")


if __name__ == "__main__":
    main()
