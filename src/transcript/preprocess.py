from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pysrt

from common import SampleIndex, annotation_path, feature_path, iter_samples, load_config


def parse_args():
    p = argparse.ArgumentParser(description="Build aligned transcript features.")
    p.add_argument("--config", type=str, default="transcript_next/config.yaml")
    return p.parse_args()


def _find_srt_file(srt_dir: Path, subject: int, story: int) -> Path | None:
    candidates = [
        srt_dir / f"transcribed_subject_{subject}_story_{story}.srt",
        srt_dir / f"transcribed_subject_{subject}_Story_{story}.srt",
        srt_dir / f"transcribed_subject_{subject}_story{story}.srt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _time_to_seconds(ts: str) -> float:
    h, m, sec_ms = ts.split(":")
    s, ms = sec_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _words_with_valence(srt_file: Path, gt_valence: np.ndarray) -> list[str]:
    subs = pysrt.open(str(srt_file))
    if len(subs) == 0:
        return []

    last_end = _time_to_seconds(str(subs[-1].end))
    fps = 25.0 if last_end <= 0 else len(gt_valence) / max(last_end, 1e-6)
    words: list[str] = []

    for sub in subs:
        cleaned = re.sub(r"[^\w\s]", "", sub.text.strip())
        tokens = cleaned.split()
        if not tokens:
            continue
        start_frame = max(0, int(_time_to_seconds(str(sub.start)) * fps))
        end_frame = min(len(gt_valence), int(_time_to_seconds(str(sub.end)) * fps))
        if end_frame <= start_frame:
            continue
        frames_per_word = (end_frame - start_frame) / max(len(tokens), 1)
        for i, token in enumerate(tokens):
            w_start = start_frame + int(i * frames_per_word)
            w_end = min(len(gt_valence), start_frame + int((i + 1) * frames_per_word))
            if w_end > w_start:
                words.append(token)
    return words


def _lexicon_lookup(words: list[str], warriner: pd.DataFrame, depeche: pd.DataFrame) -> np.ndarray:
    cols_w = ["V.Mean.Sum", "A.Mean.Sum", "D.Mean.Sum"]
    cols_d = ["AFRAID", "AMUSED", "ANGRY", "ANNOYED", "DONT_CARE", "HAPPY", "INSPIRED", "SAD"]
    prev_w = np.zeros(len(cols_w), dtype=np.float32)
    prev_d = np.zeros(len(cols_d), dtype=np.float32)
    feats: list[np.ndarray] = []

    for w in words:
        q1 = warriner.loc[warriner["Word"] == w]
        q2 = depeche.loc[depeche["Unnamed: 0"] == w]
        if len(q1) == 1:
            prev_w = np.array([float(q1.iloc[0][c]) for c in cols_w], dtype=np.float32)
        if len(q2) == 1:
            prev_d = np.array([float(q2.iloc[0][c]) for c in cols_d], dtype=np.float32)
        feats.append(np.concatenate([prev_w, prev_d], axis=0))

    if not feats:
        return np.zeros((1, 11), dtype=np.float32)
    return np.asarray(feats, dtype=np.float32)


def _upsample_to_frames(features: np.ndarray, num_frames: int) -> np.ndarray:
    num_words = len(features)
    aligned = np.zeros((num_frames, features.shape[1]), dtype=np.float32)
    base = num_frames // num_words
    rem = num_frames % num_words
    idx = 0
    for i in range(num_words):
        n = base + (1 if i < rem else 0)
        aligned[idx : idx + n] = features[i]
        idx += n
    if idx < num_frames:
        aligned[idx:] = np.mean(features, axis=0)
    return aligned


def process_sample(cfg: dict, sample: SampleIndex, warriner: pd.DataFrame, depeche: pd.DataFrame) -> bool:
    ann = annotation_path(cfg, sample)
    if not ann.exists():
        return False

    srt_file = _find_srt_file(Path(cfg["paths"]["srt_dir"]), sample.subject, sample.story)
    if srt_file is None:
        return False

    gt = pd.read_csv(ann).iloc[:, 0].to_numpy(dtype=np.float32)
    words = _words_with_valence(srt_file, gt)
    feats = _lexicon_lookup(words, warriner, depeche)
    aligned = _upsample_to_frames(feats, len(gt))

    out_path = feature_path(cfg, sample)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, aligned)
    return True


def main():
    args = parse_args()
    cfg = load_config(args.config)

    lex_dir = Path(cfg["paths"]["lexicon_dir"])
    warriner = pd.read_csv(lex_dir / "Ratings_Warriner_et_al.csv")
    depeche = pd.read_csv(lex_dir / "DepecheMood_english_token_full.tsv", delimiter="\t")

    all_samples = iter_samples(cfg, "train") + iter_samples(cfg, "val")
    # De-duplicate if train/val overlap by accident.
    unique = {(s.subject, s.story): s for s in all_samples}

    built = 0
    skipped = 0
    for sample in unique.values():
        ok = process_sample(cfg, sample, warriner, depeche)
        if ok:
            built += 1
        else:
            skipped += 1
    print(f"Done. Built {built} aligned features, skipped {skipped}.")


if __name__ == "__main__":
    main()
