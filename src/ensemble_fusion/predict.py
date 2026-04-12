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
        y_pred, _ = predict_sample(cfg, sample, ckpt_path=ckpt_path)
        out_path = write_prediction_parquet(cfg, sample, y_pred)
        validate_prediction_parquet(out_path)
        print(f"Wrote {out_path}")
        built += 1

    print(f"Done. Wrote {built} parquet files.")


if __name__ == "__main__":
    main()
