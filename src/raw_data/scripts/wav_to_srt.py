"""
Transcribe test-set WAV files to SRT using OpenAI Whisper.

Input  : src/raw_data/testing/audio/Subject_{S}_Story_{N}.mp4.wav
Output : src/raw_data/transcript/srt/transcribed_subject_{s}_story_{n}.srt

Run from the repo root:
    python src/raw_data/scripts/wav_to_srt.py [--model small] [--force]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pysrt
import whisper


TEST_SUBJECTS = list(range(1, 11))
TEST_STORIES = [3, 6, 7]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transcribe test WAVs to SRT.")
    p.add_argument(
        "--model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: small)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-transcribe even if the SRT already exists",
    )
    p.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("raw_data/testing/audio"),
        help="Directory containing test WAV files",
    )
    p.add_argument(
        "--srt-dir",
        type=Path,
        default=Path("raw_data/transcript/srt"),
        help="Directory to write SRT files into",
    )
    return p.parse_args()


def seconds_to_srt_time(seconds: float) -> pysrt.SubRipTime:
    return pysrt.SubRipTime(seconds=seconds)


def transcribe_to_srt(
    model: whisper.Whisper,
    wav_path: Path,
    srt_path: Path,
) -> None:
    print(f"  Transcribing {wav_path.name} ...")
    result = model.transcribe(str(wav_path), language="en")

    subs = pysrt.SubRipFile()
    for i, seg in enumerate(result["segments"]):
        item = pysrt.SubRipItem(
            index=i + 1,
            start=seconds_to_srt_time(seg["start"]),
            end=seconds_to_srt_time(seg["end"]),
            text=seg["text"].strip(),
        )
        subs.append(item)

    srt_path.parent.mkdir(parents=True, exist_ok=True)
    subs.save(str(srt_path), encoding="utf-8")
    print(f"  Saved -> {srt_path}")


def main() -> None:
    args = parse_args()

    print(f"Loading Whisper model '{args.model}' ...")
    model = whisper.load_model(args.model)

    skipped = 0
    processed = 0

    for subject in TEST_SUBJECTS:
        for story in TEST_STORIES:
            wav_path = args.audio_dir / f"Subject_{subject}_Story_{story}.mp4.wav"
            srt_path = args.srt_dir / f"transcribed_subject_{subject}_story_{story}.srt"

            if not wav_path.exists():
                print(f"[WARN] WAV not found, skipping: {wav_path}")
                continue

            if srt_path.exists() and not args.force:
                print(f"[SKIP] Already exists: {srt_path.name}")
                skipped += 1
                continue

            transcribe_to_srt(model, wav_path, srt_path)
            processed += 1

    print(f"\nDone. Processed: {processed}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
