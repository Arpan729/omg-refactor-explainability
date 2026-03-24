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

try:
    from common import SampleIndex, build_modality_series_for_sample, ccc_numpy, load_config
except ImportError:  # pragma: no cover
    from late_fusion.common import SampleIndex, build_modality_series_for_sample, ccc_numpy, load_config


PREDICTION_NAME_RE = re.compile(r"Subject_(\d+)_Story_(\d+)\.parquet$")
REQUIRED_COLUMNS = ["frame_idx", "y_pred", "subject_id", "story_id"]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate late-fusion prediction parquet files with CCC and plots.")
    p.add_argument("--config", type=str, default="late_fusion/config.yaml")
    p.add_argument("--output-dir", type=str, default="late_fusion/artifacts/model_evaluation")
    p.add_argument("--max-plots", type=int, default=10)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _parse_sample_from_filename(path: Path) -> tuple[int, int] | None:
    m = PREDICTION_NAME_RE.search(path.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {
            "ccc": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "pred_mean": float("nan"),
            "true_mean": float("nan"),
            "pred_std": float("nan"),
            "true_std": float("nan"),
        }
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


def align_predictions(pred_df: pd.DataFrame, y_true_full: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    warnings: list[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in pred_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame_idx = pred_df["frame_idx"].to_numpy(dtype=np.int64)
    y_pred = pred_df["y_pred"].to_numpy(dtype=np.float64)

    if pd.Index(frame_idx).duplicated().any() or not pd.Index(frame_idx).is_monotonic_increasing:
        raise ValueError("Invalid frame_idx: duplicates or not monotonic increasing")

    finite_mask = np.isfinite(y_pred)
    if not np.all(finite_mask):
        warnings.append(f"dropped_non_finite={int(np.sum(~finite_mask))}")
    frame_mask = (frame_idx >= 0) & (frame_idx < len(y_true_full))
    if not np.all(frame_mask):
        warnings.append(f"dropped_out_of_range={int(np.sum(~frame_mask))}")

    keep = finite_mask & frame_mask
    frame_idx = frame_idx[keep]
    y_pred = y_pred[keep]
    y_true = y_true_full[frame_idx]

    finite_true = np.isfinite(y_true)
    if not np.all(finite_true):
        warnings.append(f"dropped_non_finite_gt={int(np.sum(~finite_true))}")
        frame_idx = frame_idx[finite_true]
        y_pred = y_pred[finite_true]
        y_true = y_true[finite_true]

    return frame_idx.astype(np.int64), y_true.astype(np.float64), y_pred.astype(np.float64), warnings


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise RuntimeError(f"Output directory is not empty: {output_dir}. Use --overwrite to replace files.")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)


def _plot_ccc_bar(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    plot_df = df.sort_values("ccc", ascending=True)
    labels = [f"S{int(r.subject_id)}-T{int(r.story_id)}" for r in plot_df.itertuples()]

    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.35)))
    ax.barh(labels, plot_df["ccc"].to_numpy(dtype=float), color="#5B8E7D")
    ax.set_xlabel("CCC")
    ax.set_ylabel("Sample")
    ax.set_title("Late Fusion CCC by Sample")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    if len(y_true) == 0:
        return
    max_points = 20000
    if len(y_true) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(y_true), size=max_points, replace=False)
        yt = y_true[idx]
        yp = y_pred[idx]
    else:
        yt = y_true
        yp = y_pred

    vmin = float(min(np.min(yt), np.min(yp)))
    vmax = float(max(np.max(yt), np.max(yp)))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yt, yp, s=4, alpha=0.2, color="#2A9D8F")
    ax.plot([vmin, vmax], [vmin, vmax], color="#E76F51", linewidth=1.5)
    ax.set_xlabel("Ground Truth")
    ax.set_ylabel("Prediction")
    ax.set_title("Late Fusion Prediction vs Ground Truth")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


MODALITY_COLORS = {
    "speech": "#4C78A8",
    "raw_face": "#F58518",
    "transcript": "#54A24B",
    "landmarks": "#B279A2",
    "fullbody": "#9C755F",
}


def _plot_timeseries(
    frame_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    overlay_series: dict[str, np.ndarray],
    out_path: Path,
    subject_id: int,
    story_id: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(frame_idx, y_true, label="y_true", linewidth=1.0, color="#1D3557")
    for modality_name, modality_series in overlay_series.items():
        ax.plot(
            frame_idx,
            modality_series,
            label=modality_name,
            linewidth=0.9,
            alpha=0.7,
            color=MODALITY_COLORS.get(modality_name),
        )
    ax.plot(frame_idx, y_pred, label="y_pred", linewidth=1.0, color="#E63946", alpha=0.85)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Valence")
    ax.set_title(f"Late Fusion Time Series: Subject {subject_id}, Story {story_id}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_evaluation(cfg: dict[str, Any], output_dir: Path, max_plots: int, overwrite: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    _prepare_output_dir(output_dir, overwrite=overwrite)

    pred_dir = Path(cfg["paths"]["prediction_dir"])
    ann_dir = Path(cfg["paths"]["val_ann_dir"])
    prediction_files = sorted(pred_dir.glob("*.parquet"))

    rows: list[dict[str, Any]] = []
    full_true: list[np.ndarray] = []
    full_pred: list[np.ndarray] = []
    series_for_plot: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]] = []
    skipped = 0

    for path in prediction_files:
        sample = _parse_sample_from_filename(path)
        if sample is None:
            skipped += 1
            continue
        subject_id, story_id = sample
        ann_path = ann_dir / f"Subject_{subject_id}_Story_{story_id}.csv"
        if not ann_path.exists():
            rows.append(
                {
                    "subject_id": subject_id,
                    "story_id": story_id,
                    "n_points": 0,
                    "ccc": float("nan"),
                    "mae": float("nan"),
                    "rmse": float("nan"),
                    "pred_mean": float("nan"),
                    "true_mean": float("nan"),
                    "pred_std": float("nan"),
                    "true_std": float("nan"),
                    "warnings": "missing_annotation",
                }
            )
            skipped += 1
            continue

        pred_df = pd.read_parquet(path)
        y_true_full = pd.read_csv(ann_path).iloc[:, 0].to_numpy(dtype=np.float64)

        try:
            frame_idx, y_true, y_pred, warnings = align_predictions(pred_df, y_true_full)
        except ValueError as exc:
            rows.append(
                {
                    "subject_id": subject_id,
                    "story_id": story_id,
                    "n_points": 0,
                    "ccc": float("nan"),
                    "mae": float("nan"),
                    "rmse": float("nan"),
                    "pred_mean": float("nan"),
                    "true_mean": float("nan"),
                    "pred_std": float("nan"),
                    "true_std": float("nan"),
                    "warnings": str(exc),
                }
            )
            skipped += 1
            continue

        metrics = _compute_metrics(y_true, y_pred)
        rows.append(
            {
                "subject_id": subject_id,
                "story_id": story_id,
                "n_points": int(len(y_true)),
                **metrics,
                "warnings": ";".join(warnings),
            }
        )

        if len(y_true) > 0:
            full_true.append(y_true)
            full_pred.append(y_pred)
            if len(series_for_plot) < max_plots:
                overlay_series = build_modality_series_for_sample(
                    cfg,
                    SampleIndex(subject=subject_id, story=story_id, split="val"),
                )
                series_for_plot.append((subject_id, story_id, frame_idx, y_true, y_pred, overlay_series))

    metrics_df = pd.DataFrame(rows).sort_values(["subject_id", "story_id"]).reset_index(drop=True)
    metrics_df.to_csv(output_dir / "metrics_per_sample.csv", index=False)

    overall_true = np.concatenate(full_true) if full_true else np.array([], dtype=np.float64)
    overall_pred = np.concatenate(full_pred) if full_pred else np.array([], dtype=np.float64)
    overall = _compute_metrics(overall_true, overall_pred)
    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "n_prediction_files": int(len(prediction_files)),
        "n_samples_evaluated": int(len(metrics_df)),
        "n_samples_skipped": int(skipped),
        "overall_ccc": overall["ccc"],
        "overall_mae": overall["mae"],
        "overall_rmse": overall["rmse"],
        "mean_sample_ccc": float(metrics_df["ccc"].dropna().mean()) if not metrics_df.empty else float("nan"),
    }
    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _plot_ccc_bar(metrics_df, output_dir / "plots" / "ccc_by_sample.png")
    _plot_scatter(overall_true, overall_pred, output_dir / "plots" / "distribution_scatter.png")
    for subject_id, story_id, frame_idx, y_true, y_pred, overlay_series in series_for_plot:
        out = output_dir / "plots" / f"timeseries_subject_{subject_id}_story_{story_id}.png"
        _plot_timeseries(frame_idx, y_true, y_pred, overlay_series, out, subject_id, story_id)

    return metrics_df, summary


def main():
    args = parse_args()
    cfg = load_config(args.config)
    metrics_df, summary = run_evaluation(cfg, Path(args.output_dir), max_plots=args.max_plots, overwrite=args.overwrite)
    print(metrics_df.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
