"""Generate out-of-fold (OOF) predictions for landmarks model (story-level CV on training stories)."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd

try:
    from common import load_config, set_seed
    from predict import predict_samples
    from train import train_model
except ImportError:  # pragma: no cover
    from landmarks.common import load_config, set_seed
    from landmarks.predict import predict_samples
    from landmarks.train import train_model


def parse_args():
    p = argparse.ArgumentParser(description="Train holdout models and write OOF predictions for landmarks.")
    p.add_argument("--config", type=str, default="landmarks/config.yaml")
    return p.parse_args()


def combine_oof_parquets(oof_dir: Path) -> Path:
    """Merge per-sample OOF parquets into combined_oof.parquet."""
    oof_dir = Path(oof_dir)
    parquets = sorted(oof_dir.glob("Subject_*_Story_*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No OOF parquet files found under {oof_dir}")
    dfs = [pd.read_parquet(p) for p in parquets]
    combined = pd.concat(dfs, ignore_index=True)
    out_path = oof_dir / "combined_oof.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} ({len(combined)} rows)")
    return out_path


def main():
    import torch

    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["train"]["seed"]))

    folds = list(cfg["split"]["stories_train"])
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    oof_dir = Path(cfg["paths"]["prediction_dir"]) / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)

    oof_cfg = copy.deepcopy(cfg)
    oof_cfg["paths"]["val_ann_dir"] = oof_cfg["paths"]["train_ann_dir"]

    for holdout in folds:
        print(f"\n{'=' * 50}")
        print(f"Fold: holdout story {holdout}")
        print(f"{'=' * 50}")

        train_stories = [s for s in folds if s != holdout]
        ckpt_path = ckpt_dir / f"landmarks_conv1d_holdout_{holdout}.pt"

        best_ccc = train_model(oof_cfg, train_stories, [holdout], ckpt_path)
        print(f"Holdout {holdout} best val CCC: {best_ccc:.6f}")

        predict_samples(
            oof_cfg,
            ckpt_path,
            [holdout],
            oof_dir,
            split="train",
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    combine_oof_parquets(oof_dir)
    print("\nOOF generation complete.")


if __name__ == "__main__":
    main()
