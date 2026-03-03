from __future__ import annotations

import argparse

import numpy as np

from common import (
    LandmarksConv1DModel,
    SampleIndex,
    checkpoint_path,
    choose_device,
    denorm_target,
    iter_samples,
    load_aligned,
    load_config,
    validate_prediction_parquet,
    window_landmarks,
    write_prediction_parquet,
)


def parse_args():
    p = argparse.ArgumentParser(description="Predict landmarks val outputs to parquet.")
    p.add_argument("--config", type=str, default="landmarks/config.yaml")
    return p.parse_args()


def predict_frames(model, x: np.ndarray, batch_size: int, device: "torch.device") -> np.ndarray:
    import torch

    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = start + batch_size
            xb = torch.tensor(x[start:end], dtype=torch.float32).to(device)
            yb = model(xb)
            outputs.append(yb.detach().cpu().numpy().reshape(-1))
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
    feature_mean = float(saved["feature_mean"])
    feature_std = float(saved["feature_std"])
    target_min = float(saved["target_min"])
    target_max = float(saved["target_max"])

    model = LandmarksConv1DModel(cfg).to(device)
    model.load_state_dict(saved["model_state"])

    window_size = int(cfg["model"]["window_size"])

    built = 0
    for sample in iter_samples(cfg, "val"):
        sample = SampleIndex(subject=sample.subject, story=sample.story, split="val")
        try:
            x, y = load_aligned(cfg, sample)
        except FileNotFoundError as exc:
            print(f"Skipping {sample.subject}/{sample.story}: {exc}")
            continue

        n = min(len(x), len(y))
        if n == 0:
            continue
        xw = window_landmarks(x[:n], window_size)
        xw = (xw - feature_mean) / max(feature_std, 1e-8)

        preds = predict_frames(model, xw, batch_size, device)
        preds = denorm_target(preds, target_min, target_max)

        out_path = write_prediction_parquet(cfg, sample, preds.astype(np.float32))
        validate_prediction_parquet(out_path)
        print(f"Wrote {out_path}")
        built += 1

    print(f"Done. Wrote {built} parquet files.")


if __name__ == "__main__":
    main()
