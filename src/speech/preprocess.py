from __future__ import annotations

import argparse

from common import build_aligned_features, iter_samples, load_config


def parse_args():
    p = argparse.ArgumentParser(description="Build aligned speech features with PyTorch STFT.")
    p.add_argument("--config", type=str, default="speech/config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    all_samples = iter_samples(cfg, "train") + iter_samples(cfg, "val")
    unique = {(s.subject, s.story): s for s in all_samples}

    built = 0
    skipped = 0
    for sample in unique.values():
        try:
            out = build_aligned_features(cfg, sample)
            print(f"Built {out}")
            built += 1
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping {sample.subject}/{sample.story}: {exc}")
            skipped += 1

    print(f"Done. Built {built} aligned feature files, skipped {skipped}.")


if __name__ == "__main__":
    main()
