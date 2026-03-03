from __future__ import annotations

import argparse

import cv2
import dlib
import numpy as np

from common import iter_samples, landmark_csv_path, load_config, video_path


def parse_args():
    p = argparse.ArgumentParser(description="Extract subject landmark CSVs directly from videos.")
    p.add_argument("--config", type=str, default="landmarks/config.yaml")
    return p.parse_args()


def _pick_largest_rect(dets):
    if not dets:
        return None
    return max(dets, key=lambda d: max(0, d.right() - d.left()) * max(0, d.bottom() - d.top()))


def _shape_to_np(shape) -> np.ndarray:
    coords = np.zeros((68, 2), dtype=np.int32)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def extract_subject_landmarks_for_sample(cfg: dict, sample, detector, predictor) -> bool:
    vp = video_path(cfg, sample)
    if not vp.exists():
        return False

    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        return False

    upsample = int(cfg["extract"].get("detector_upsample", 1))
    landmarks_rows = []
    prev = None

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        h, w = frame.shape[:2]
        mid = w // 2
        subject_half = frame[:, mid:]

        det = _pick_largest_rect(detector(subject_half, upsample))
        if det is not None:
            try:
                shape = predictor(subject_half, det)
                lm = _shape_to_np(shape)
                # Convert half-frame coordinates back to full-frame coordinates.
                lm = lm + np.array([mid, 0], dtype=np.int32)
                prev = lm.reshape(-1)
                landmarks_rows.append(prev)
                continue
            except Exception:
                pass

        if prev is not None:
            landmarks_rows.append(prev)
        else:
            landmarks_rows.append(np.zeros((136,), dtype=np.int32))

    cap.release()

    if not landmarks_rows:
        return False

    out = landmark_csv_path(cfg, sample)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, np.asarray(landmarks_rows, dtype=np.int32), fmt="%d", delimiter=",")
    return True


def main():
    args = parse_args()
    cfg = load_config(args.config)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(cfg["paths"]["predictor_path"]))

    built = 0
    skipped = 0
    for split in ["train", "val"]:
        for sample in iter_samples(cfg, split):
            ok = extract_subject_landmarks_for_sample(cfg, sample, detector, predictor)
            if ok:
                built += 1
                print(f"Built Subject landmarks CSV for {sample.subject}/{sample.story} ({sample.split})")
            else:
                skipped += 1
                print(f"Skipped {sample.subject}/{sample.story} ({sample.split})")

    print(f"Done. Built {built} landmark CSV files, skipped {skipped}.")


if __name__ == "__main__":
    main()
