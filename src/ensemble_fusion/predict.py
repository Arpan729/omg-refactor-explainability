from __future__ import annotations

import argparse

try:
    from common import checkpoint_path, iter_samples, load_config, predict_sample, validate_prediction_parquet, write_prediction_parquet
except ImportError:  # pragma: no cover
    from ensemble_fusion.common import (
        checkpoint_path,
        iter_samples,
        load_config,
        predict_sample,
        validate_prediction_parquet,
        write_prediction_parquet,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Predict ensemble-fusion val outputs to parquet.")
    p.add_argument("--config", type=str, default="ensemble_fusion/config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    ckpt_path = checkpoint_path(cfg)

    built = 0
    for sample in iter_samples(cfg, "val"):
        y_pred_raw, y_pred_smoothed, _ = predict_sample(cfg, sample, ckpt_path=ckpt_path)
        out_raw = write_prediction_parquet(cfg, sample, y_pred_raw, variant="raw")
        out_smoothed = write_prediction_parquet(cfg, sample, y_pred_smoothed, variant="smoothed")
        validate_prediction_parquet(out_raw)
        validate_prediction_parquet(out_smoothed)
        print(f"Wrote {out_raw}")
        print(f"Wrote {out_smoothed}")
        built += 1

    print(f"Done. Wrote {built} raw and {built} smoothed parquet files.")


if __name__ == "__main__":
    main()
