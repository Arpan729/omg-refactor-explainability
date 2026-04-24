from __future__ import annotations

import argparse

try:
    from common import (
        checkpoint_path,
        cv_fold_metrics_csv_path,
        load_config,
        sweep_summary_csv_path,
        sweep_summary_json_path,
        train_model,
        train_model_sweep,
    )
except ImportError:  # pragma: no cover
    from ensemble_fusion.common import (
        checkpoint_path,
        cv_fold_metrics_csv_path,
        load_config,
        sweep_summary_csv_path,
        sweep_summary_json_path,
        train_model,
        train_model_sweep,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Train ensemble fusion nonnegative elastic-net stacker.")
    p.add_argument("--config", type=str, default="ensemble_fusion/config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    mode = str(cfg["train"].get("mode", "single"))
    if mode == "sweep":
        ckpt_path, summary, summary_df = train_model_sweep(cfg)
        print(f"Saved best checkpoint: {ckpt_path}")
        print(f"Sweep summary CSV: {sweep_summary_csv_path(cfg)}")
        print(f"CV fold metrics CSV: {cv_fold_metrics_csv_path(cfg)}")
        print(f"Sweep summary JSON: {sweep_summary_json_path(cfg)}")
        print(summary_df.to_string(index=False))
        print(f"Selected run: {summary['best_run_name']}")
        print(f"Selection variant: {summary['selection_variant']}")
        print(f"Selection protocol: {summary['selection_protocol']}")
        return

    ckpt_path, metrics = train_model(cfg, checkpoint_out=checkpoint_path(cfg))
    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Model: {metrics['model_name']}")
    print(f"Train CCC: {metrics['train_ccc']:.6f}")
    print(f"Train RMSE: {metrics['train_rmse']:.6f}")


if __name__ == "__main__":
    main()
