from __future__ import annotations

import argparse

import numpy as np

from common import (
    TranscriptLSTMModel,
    TranscriptWindowDataset,
    ccc_loss_torch,
    ccc_numpy,
    checkpoint_path,
    choose_device,
    load_config,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train transcript_next PyTorch model.")
    p.add_argument("--config", type=str, default="transcript_next/config.yaml")
    return p.parse_args()


def evaluate(model, loader, device):
    import torch

    model.eval()
    y_true_all = []
    y_pred_all = []
    with torch.no_grad():
        for x, sid, y in loader:
            pred = model(x.to(device), sid.to(device))
            y_true_all.append(y.numpy())
            y_pred_all.append(pred.detach().cpu().numpy())
    if not y_true_all:
        return float("nan")
    return ccc_numpy(np.concatenate(y_true_all), np.concatenate(y_pred_all))


def main():
    import torch
    from torch.utils.data import DataLoader

    args = parse_args()
    cfg = load_config(args.config)
    set_seed(42)

    train_cfg = cfg["train"]
    device = choose_device(str(train_cfg["device"]))
    print(f"Using device: {device}")

    train_ds = TranscriptWindowDataset(cfg, split="train")
    if len(train_ds) == 0:
        raise RuntimeError("No training windows found. Run preprocess first.")
    val_ds = TranscriptWindowDataset(
        cfg,
        split="val",
        label_min=train_ds.label_min,
        label_max=train_ds.label_max,
    )

    train_loader = DataLoader(train_ds, batch_size=int(train_cfg["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(train_cfg["batch_size"]), shuffle=False)

    model = TranscriptLSTMModel(cfg).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=float(train_cfg["lr"]))

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
            loss = ccc_loss_torch(y, pred)
            loss.backward()
            optim.step()
            total_loss += float(loss.item())

        train_loss = total_loss / max(len(train_loader), 1)
        val_ccc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_ccc={val_ccc:.6f}")

        if val_ccc > best_ccc:
            best_ccc = val_ccc
            no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "label_min": train_ds.label_min,
                    "label_max": train_ds.label_max,
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
