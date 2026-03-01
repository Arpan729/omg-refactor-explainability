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
    from common import ccc_numpy, load_config
except ImportError:  # pragma: no cover
    from transcript.common import ccc_numpy, load_config


PREDICTION_NAME_RE = re.compile(r"Subject_(\d+)_Story_(\d+)\.parquet$")
REQUIRED_COLUMNS = [
    "window_idx",
    "window_start_frame",
    "window_end_frame",
    "y_pred",
    "subject_id",
    "story_id",
]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate transcript prediction parquet files with CCC and plots.")
    p.add_argument("--config", type=str, default="transcript/config.yaml")
    p.add_argument("--output-dir", type=str, default="transcript/artifacts/model_evaluation")
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


def align_transcript_predictions(
    pred_df: pd.DataFrame, y_true_full: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    warnings: list[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in pred_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    window_idx = pred_df["window_idx"].to_numpy(dtype=np.int64)
    starts = pred_df["window_start_frame"].to_numpy(dtype=np.int64)
    ends = pred_df["window_end_frame"].to_numpy(dtype=np.int64)
    y_pred = pred_df["y_pred"].to_numpy(dtype=np.float64)

    if pd.Index(window_idx).duplicated().any() or not pd.Index(window_idx).is_monotonic_increasing:
        raise ValueError("Invalid window_idx: duplicates or not monotonic increasing")

    keep_window_idx: list[int] = []
    keep_true: list[float] = []
    keep_pred: list[float] = []

    for idx, start, end, pred in zip(window_idx, starts, ends, y_pred):
        if not np.isfinite(pred):
            warnings.append(f"window_{idx}:non_finite_pred")
            continue
        start_clipped = max(0, int(start))
        end_clipped = min(int(end), len(y_true_full) - 1)
        if end_clipped < start_clipped:
            warnings.append(f"window_{idx}:invalid_bounds")
            continue
        y_segment = y_true_full[start_clipped : end_clipped + 1]
        if len(y_segment) == 0:
            warnings.append(f"window_{idx}:empty_segment")
            continue
        y_mean = float(np.mean(y_segment))
        if not np.isfinite(y_mean):
            warnings.append(f"window_{idx}:non_finite_gt")
            continue

        keep_window_idx.append(int(idx))
        keep_true.append(y_mean)
        keep_pred.append(float(pred))

    return (
        np.asarray(keep_window_idx, dtype=np.int64),
        np.asarray(keep_true, dtype=np.float64),
        np.asarray(keep_pred, dtype=np.float64),
        warnings,
    )


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
    ax.barh(labels, plot_df["ccc"].to_numpy(dtype=float), color="#3A86FF")
    ax.set_xlabel("CCC")
    ax.set_ylabel("Sample")
    ax.set_title("Transcript CCC by Sample")
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
    ax.scatter(yt, yp, s=8, alpha=0.22, color="#2A9D8F")
    ax.plot([vmin, vmax], [vmin, vmax], color="#E76F51", linewidth=1.5)
    ax.set_xlabel("Ground Truth Window Mean")
    ax.set_ylabel("Prediction")
    ax.set_title("Transcript Prediction vs Ground Truth")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_window_series(
    window_idx: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, subject_id: int, story_id: int
) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(window_idx, y_true, label="y_true_window_mean", linewidth=1.0, color="#1D3557")
    ax.plot(window_idx, y_pred, label="y_pred", linewidth=1.0, color="#E63946", alpha=0.85)
    ax.set_xlabel("Window Index")
    ax.set_ylabel("Valence")
    ax.set_title(f"Transcript Window Series: Subject {subject_id}, Story {story_id}")
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
    series_for_plot: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = []
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
            window_idx, y_true, y_pred, warnings = align_transcript_predictions(pred_df, y_true_full)
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
        row = {
            "subject_id": subject_id,
            "story_id": story_id,
            "n_points": int(len(y_true)),
            **metrics,
            "warnings": ";".join(warnings),
        }
        rows.append(row)

        if len(y_true) > 0:
            full_true.append(y_true)
            full_pred.append(y_pred)
            series_for_plot.append((subject_id, story_id, window_idx, y_true, y_pred))

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "metrics_per_sample.csv", index=False)

    if full_true:
        global_true = np.concatenate(full_true)
        global_pred = np.concatenate(full_pred)
        overall_ccc = float(ccc_numpy(global_true, global_pred))
    else:
        global_true = np.asarray([], dtype=np.float64)
        global_pred = np.asarray([], dtype=np.float64)
        overall_ccc = float("nan")

    valid_ccc = metrics_df["ccc"].dropna() if "ccc" in metrics_df else pd.Series(dtype=float)
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "num_prediction_files": len(prediction_files),
        "num_samples_evaluated": int(len(metrics_df) - skipped),
        "num_samples_skipped": int(skipped),
        "overall_ccc": overall_ccc,
        "mean_sample_ccc": float(valid_ccc.mean()) if not valid_ccc.empty else float("nan"),
        "median_sample_ccc": float(valid_ccc.median()) if not valid_ccc.empty else float("nan"),
    }
    if not valid_ccc.empty:
        best_idx = metrics_df["ccc"].idxmax()
        worst_idx = metrics_df["ccc"].idxmin()
        summary["best_sample"] = {
            "subject_id": int(metrics_df.loc[best_idx, "subject_id"]),
            "story_id": int(metrics_df.loc[best_idx, "story_id"]),
            "ccc": float(metrics_df.loc[best_idx, "ccc"]),
        }
        summary["worst_sample"] = {
            "subject_id": int(metrics_df.loc[worst_idx, "subject_id"]),
            "story_id": int(metrics_df.loc[worst_idx, "story_id"]),
            "ccc": float(metrics_df.loc[worst_idx, "ccc"]),
        }

    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plots_dir = output_dir / "plots"
    _plot_ccc_bar(metrics_df.dropna(subset=["ccc"]), plots_dir / "ccc_by_sample.png")
    _plot_scatter(global_true, global_pred, plots_dir / "distribution_scatter.png")

    for subject_id, story_id, window_idx, y_true, y_pred in series_for_plot[: max(0, max_plots)]:
        out = plots_dir / f"window_series_subject_{subject_id}_story_{story_id}.png"
        _plot_window_series(window_idx, y_true, y_pred, out, subject_id=subject_id, story_id=story_id)

    return metrics_df, summary


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = Path(args.output_dir).resolve()
    metrics_df, summary = run_evaluation(
        cfg=cfg,
        output_dir=output_dir,
        max_plots=int(args.max_plots),
        overwrite=bool(args.overwrite),
    )

    print(f"Wrote metrics: {output_dir / 'metrics_per_sample.csv'}")
    print(f"Wrote summary: {output_dir / 'metrics_summary.json'}")
    print(f"Wrote plots dir: {output_dir / 'plots'}")
    print(
        "Summary | "
        f"samples={len(metrics_df)} "
        f"overall_ccc={summary.get('overall_ccc', float('nan')):.6f} "
        f"mean_sample_ccc={summary.get('mean_sample_ccc', float('nan')):.6f}"
    )


if __name__ == "__main__":
    main()
