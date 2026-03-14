from __future__ import annotations

import argparse

try:
    from common import fuse_sample_predictions, iter_samples, load_config, validate_prediction_parquet, write_prediction_parquet
except ImportError:  # pragma: no cover
    from late_fusion.common import (
        fuse_sample_predictions,
        iter_samples,
        load_config,
        validate_prediction_parquet,
        write_prediction_parquet,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Predict late-fusion val outputs to parquet.")
    p.add_argument("--config", type=str, default="late_fusion/config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    built = 0
    for sample in iter_samples(cfg, "val"):
        try:
            y_pred, _ = fuse_sample_predictions(cfg, sample)
        except FileNotFoundError as exc:
            print(f"Skipping {sample.subject}/{sample.story}: {exc}")
            continue

        out_path = write_prediction_parquet(cfg, sample, y_pred)
        validate_prediction_parquet(out_path)
        print(f"Wrote {out_path}")
        built += 1

    print(f"Done. Wrote {built} parquet files.")


if __name__ == "__main__":
    main()
