from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from common import (
    TranscriptLSTMModel,
    checkpoint_path,
    choose_device,
    iter_samples_for_stories,
    load_config,
    read_features,
    validate_prediction_parquet,
    window_features,
    write_prediction_parquet,
)

if TYPE_CHECKING:
    import torch


def parse_args():
    p = argparse.ArgumentParser(description="Predict transcript_next val outputs to parquet.")
    p.add_argument("--config", type=str, default="transcript/config.yaml")
    return p.parse_args()


def predict_windows(model, x_windows: np.ndarray, subject_id: int, batch_size: int, device) -> np.ndarray:
    import torch

    model.eval()
    outputs = []
    sid_tensor = torch.full((len(x_windows),), subject_id - 1, dtype=torch.long)
    with torch.no_grad():
        for start in range(0, len(x_windows), batch_size):
            end = start + batch_size
            xb = torch.tensor(x_windows[start:end], dtype=torch.float32).to(device)
            sb = sid_tensor[start:end].to(device)
            yb = model(xb, sb)
            outputs.append(yb.detach().cpu().numpy())
    return np.concatenate(outputs) if outputs else np.zeros((0,), dtype=np.float32)


def predict_samples(
    cfg: dict,
    ckpt_path: Path,
    target_stories: list[int],
    output_dir: Path,
    split: str = "val",
    device_flag: str | None = None,
) -> list[Path]:
    """Generate predictions for subject×target_stories. Returns written parquet paths."""
    import torch

    cfg = copy.deepcopy(cfg)
    device = choose_device(str(device_flag or cfg["predict"]["device"]))
    batch_size = int(cfg["predict"]["batch_size"])
    print(f"Using device: {device}")

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint missing: {ckpt_path}")

    saved = torch.load(ckpt_path, map_location=device)
    label_min = float(saved["label_min"])
    label_max = float(saved["label_max"])

    model = TranscriptLSTMModel(cfg).to(device)
    model.load_state_dict(saved["model_state"])

    out_paths: list[Path] = []
    built = 0
    for sample in iter_samples_for_stories(cfg, split, target_stories):
        try:
            x = read_features(cfg, sample)
        except FileNotFoundError as exc:
            print(f"Skipping {sample.subject}/{sample.story}: {exc}")
            continue

        xw = window_features(
            x,
            window_size=int(cfg["model"]["window_size"]),
            stride=int(cfg["model"]["stride"]),
        )
        if len(xw) == 0:
            print(f"Skipping {sample.subject}/{sample.story}: not enough frames")
            continue

        preds = predict_windows(model, xw, sample.subject, batch_size, device)
        preds = preds * (label_max - label_min + 1e-8) + label_min

        out_path = write_prediction_parquet(cfg, sample, preds.astype(np.float32), output_dir=output_dir)
        validate_prediction_parquet(out_path)
        print(f"Wrote {out_path}")
        out_paths.append(out_path)
        built += 1

    print(f"Done. Wrote {built} parquet files.")
    del model
    return out_paths


def main():
    args = parse_args()
    cfg = load_config(args.config)
    predict_samples(
        cfg,
        checkpoint_path(cfg),
        list(cfg["split"]["stories_val"]),
        Path(cfg["paths"]["prediction_dir"]),
        split="val",
    )


if __name__ == "__main__":
    main()
