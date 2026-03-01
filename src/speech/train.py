from __future__ import annotations

import argparse

import numpy as np

from common import (
    SampleIndex,
    SpeechBiGRUModel,
    SpeechWindowDataset,
    ccc_loss_torch_sequence,
    ccc_numpy,
    checkpoint_path,
    choose_device,
    denorm_target,
    iter_samples,
    load_config,
    read_features,
    read_labels,
    reconstruct_from_windows,
    set_seed,
    window_features,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train speech PyTorch BiGRU sequence model.")
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


def evaluate(
    cfg: dict,
    model,
    device,
    batch_size: int,
    feature_mean: float,
    feature_std: float,
    target_min: float,
    target_max: float,
):
    y_true_all = []
    y_pred_all = []
    seq_len = int(cfg["model"]["sequence_length"])
    stride = int(cfg["model"]["stride"])

    for sample in iter_samples(cfg, "val"):
        sample = SampleIndex(subject=sample.subject, story=sample.story, split="val")
        try:
            x = read_features(cfg, sample)
            y = read_labels(cfg, sample)
        except FileNotFoundError:
            continue

        x_norm = (x - feature_mean) / max(feature_std, 1e-8)
        xw, starts = window_features(
            x_norm,
            window_size=seq_len,
            stride=stride,
            include_last=True,
        )
        if len(xw) == 0:
            continue

        pred_windows = predict_windows(model, xw, sample.subject, batch_size, device)
        pred_windows = denorm_target(pred_windows, target_min, target_max)
        pred_frames = reconstruct_from_windows(pred_windows, starts, total_len=len(y))

        y_true_all.append(y.reshape(-1))
        y_pred_all.append(pred_frames.reshape(-1))

    if not y_true_all:
        return float("nan")
    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    return ccc_numpy(y_true, y_pred)


def main():
    import torch
    from torch.utils.data import DataLoader

    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["train"]["seed"]))

    train_cfg = cfg["train"]
    device = choose_device(str(train_cfg["device"]))
    print(f"Using device: {device}")

    train_ds = SpeechWindowDataset(cfg, split="train")
    if len(train_ds) == 0:
        raise RuntimeError("No training windows found. Run preprocess first.")

    train_loader = DataLoader(train_ds, batch_size=int(train_cfg["batch_size"]), shuffle=True)
    val_batch_size = int(train_cfg["batch_size"])

    model = SpeechBiGRUModel(cfg).to(device)
    optim = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    best_ccc = float("-inf")
    no_improve = 0
    patience = int(train_cfg["patience"])
    ckpt = checkpoint_path(cfg)

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        for x, sid, y in train_loader:
            x = x.to(device)
            sid = sid.to(device)
            y = y.to(device)

            optim.zero_grad(set_to_none=True)
            pred = model(x, sid)
            loss = ccc_loss_torch_sequence(y, pred)
            loss.backward()
            optim.step()
            total_loss += float(loss.item())

        train_loss = total_loss / max(len(train_loader), 1)
        val_ccc = evaluate(
            cfg=cfg,
            model=model,
            device=device,
            batch_size=val_batch_size,
            feature_mean=train_ds.feature_mean,
            feature_std=train_ds.feature_std,
            target_min=train_ds.target_min,
            target_max=train_ds.target_max,
        )
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_ccc={val_ccc:.6f}")

        if val_ccc > best_ccc:
            best_ccc = val_ccc
            no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_mean": train_ds.feature_mean,
                    "feature_std": train_ds.feature_std,
                    "target_min": train_ds.target_min,
                    "target_max": train_ds.target_max,
                },
                ckpt,
            )
            print(f"Saved checkpoint: {ckpt}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break


if __name__ == "__main__":
    main()
