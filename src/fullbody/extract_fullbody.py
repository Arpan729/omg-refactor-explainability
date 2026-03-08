from __future__ import annotations

import argparse

from common import extract_fullbody_for_sample, iter_samples, load_config


def parse_args():
    p = argparse.ArgumentParser(description="Extract fullbody crops (actor+subject) from videos.")
    p.add_argument("--config", type=str, default="fullbody/config.yaml")
    p.add_argument("--trial", action="store_true", help="Extract only a few preview frames per sample.")
    p.add_argument("--preview-frames", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    all_samples = iter_samples(cfg, "train") + iter_samples(cfg, "val")
    unique = {(s.subject, s.story, s.split): s for s in all_samples}

    built = 0
    skipped = 0
    preview_frames = int(args.preview_frames) if args.trial else None

    for sample in unique.values():
        n = extract_fullbody_for_sample(cfg, sample, preview_frames=preview_frames)
        if n > 0:
            print(f"Extracted {n} frames for {sample.split} Subject_{sample.subject}_Story_{sample.story}")
            built += 1
        else:
            print(f"Skipping {sample.split} Subject_{sample.subject}_Story_{sample.story}: missing/invalid video")
            skipped += 1

    print(f"Done. Extracted {built} samples, skipped {skipped}.")


if __name__ == "__main__":
    main()
