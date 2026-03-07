from __future__ import annotations

import argparse

import dlib

from common import extract_subject_faces_for_sample, iter_samples, load_config


def parse_args():
    p = argparse.ArgumentParser(description="Extract subject face crops directly from videos.")
    p.add_argument("--config", type=str, default="raw_face/config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    detector = dlib.get_frontal_face_detector()

    built = 0
    skipped = 0
    for split in ["train", "val"]:
        for sample in iter_samples(cfg, split):
            ok = extract_subject_faces_for_sample(cfg, sample, detector)
            if ok:
                built += 1
            else:
                skipped += 1
    print(f"Done. Built {built} face directories, skipped {skipped}.")


if __name__ == "__main__":
    main()
