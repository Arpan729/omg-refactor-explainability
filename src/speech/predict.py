from __future__ import annotations

import argparse

import numpy as np

from common import (
    SampleIndex,
    SpeechBiGRUModel,
    checkpoint_path,
    choose_device,
    denorm_target,
    iter_samples,
    load_config,
    read_features,
    reconstruct_from_windows,
    validate_prediction_parquet,
    window_features,
    write_prediction_parquet,
)


def parse_args():
    p = argparse.ArgumentParser(description="Predict speech val outputs to parquet.")
    p.add_argument("--config", type=str, default="speech/config.yaml")
    return p.parse_args()


def predict_windows(model, x_windows: np.ndarray, subject_id: int, batch_size: int, device: "torch.device") -> np.ndarray:
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
    if not outputs:
        return np.zeros((0, x_windows.shape[1]), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


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
    feature_mean = float(saved["feature_mean"])
    feature_std = float(saved["feature_std"])
    target_min = float(saved["target_min"])
    target_max = float(saved["target_max"])

    model = SpeechBiGRUModel(cfg).to(device)
    model.load_state_dict(saved["model_state"])

    seq_len = int(cfg["model"]["sequence_length"])
    stride = int(cfg["model"]["stride"])

    built = 0
    for sample in iter_samples(cfg, "val"):
        sample = SampleIndex(subject=sample.subject, story=sample.story, split="val")
        try:
            x = read_features(cfg, sample)
        except FileNotFoundError as exc:
            print(f"Skipping {sample.subject}/{sample.story}: {exc}")
            continue

        x_norm = (x - feature_mean) / max(feature_std, 1e-8)
        xw, starts = window_features(
            x_norm,
            window_size=seq_len,
            stride=stride,
            include_last=True,
        )
        if len(xw) == 0:
            print(f"Skipping {sample.subject}/{sample.story}: not enough frames")
            continue

        pred_windows = predict_windows(model, xw, sample.subject, batch_size, device)
        pred_windows = denorm_target(pred_windows, target_min, target_max)
        pred_frames = reconstruct_from_windows(pred_windows, starts, total_len=len(x))

        out_path = write_prediction_parquet(cfg, sample, pred_frames.astype(np.float32))
        validate_prediction_parquet(out_path)
        print(f"Wrote {out_path}")
        built += 1

    print(f"Done. Wrote {built} parquet files.")


if __name__ == "__main__":
    main()
