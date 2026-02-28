from __future__ import annotations

import argparse

import numpy as np

from common import (
    SampleIndex,
    TranscriptLSTMModel,
    checkpoint_path,
    choose_device,
    iter_samples,
    load_config,
    read_features,
    read_labels,
    validate_prediction_parquet,
    window_sequence,
    write_prediction_parquet,
)


def parse_args():
    p = argparse.ArgumentParser(description="Predict transcript_next val outputs to parquet.")
    p.add_argument("--config", type=str, default="transcript/config.yaml")
    return p.parse_args()


def predict_windows(model, x_windows: np.ndarray, subject_id: int, batch_size: int, device: torch.device) -> np.ndarray:
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


def main():
    import torch

    args = parse_args()
    cfg = load_config(args.config)

    device = choose_device(str(cfg["predict"]["device"]))
    batch_size = int(cfg["predict"]["batch_size"])
    print(f"Using device: {device}")

    ckpt = checkpoint_path(cfg)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint missing: {ckpt}")

    saved = torch.load(ckpt, map_location=device)
    label_min = float(saved["label_min"])
    label_max = float(saved["label_max"])

    model = TranscriptLSTMModel(cfg).to(device)
    model.load_state_dict(saved["model_state"])

    built = 0
    for sample in iter_samples(cfg, "val"):
        sample = SampleIndex(subject=sample.subject, story=sample.story, split="val")
        try:
            x = read_features(cfg, sample)
            y = read_labels(cfg, sample)
        except FileNotFoundError as exc:
            print(f"Skipping {sample.subject}/{sample.story}: {exc}")
            continue

        xw, _ = window_sequence(
            x,
            y,
            window_size=int(cfg["model"]["window_size"]),
            stride=int(cfg["model"]["stride"]),
        )
        if len(xw) == 0:
            print(f"Skipping {sample.subject}/{sample.story}: not enough frames")
            continue

        preds = predict_windows(model, xw, sample.subject, batch_size, device)
        preds = preds * (label_max - label_min + 1e-8) + label_min

        out_path = write_prediction_parquet(cfg, sample, preds.astype(np.float32))
        validate_prediction_parquet(out_path)
        print(f"Wrote {out_path}")
        built += 1

    print(f"Done. Wrote {built} parquet files.")


if __name__ == "__main__":
    main()
