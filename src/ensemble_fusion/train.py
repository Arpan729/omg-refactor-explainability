from __future__ import annotations

import argparse

try:
    from common import checkpoint_path, load_config, train_model
except ImportError:  # pragma: no cover
    from ensemble_fusion.common import checkpoint_path, load_config, train_model


def parse_args():
    p = argparse.ArgumentParser(description="Train ensemble fusion nonnegative elastic-net stacker.")
    p.add_argument("--config", type=str, default="ensemble_fusion/config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    ckpt_path, metrics = train_model(cfg, checkpoint_out=checkpoint_path(cfg))
    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Train CCC: {metrics['train_ccc']:.6f}")
    print(f"Train RMSE: {metrics['train_rmse']:.6f}")


if __name__ == "__main__":
    main()
