from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
import yaml

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None


REQUIRED_TOP_LEVEL_KEYS = ["paths", "split", "fusion", "predict"]
REQUIRED_PATH_KEYS = ["train_ann_dir", "val_ann_dir", "prediction_dir"]
PREDICTION_COLUMNS = ["frame_idx", "timestamp_s", "y_pred", "subject_id", "story_id", "split", "manifest_id"]
FRAME_MODALITY_COLUMNS = ["frame_idx", "y_pred", "subject_id", "story_id"]
WINDOW_MODALITY_COLUMNS = ["window_idx", "window_start_frame", "window_end_frame", "y_pred", "subject_id", "story_id"]


@dataclass(frozen=True)
class SampleIndex:
    subject: int
    story: int
    split: str


@dataclass(frozen=True)
class ModalityConfig:
    name: str
    prediction_dir: Path
    weight: float
    cutoff: float
    order: int
    kind: str


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML dictionary.")

    _validate_config(cfg)
    _resolve_paths(cfg)
    _validate_dirs(cfg)
    _ensure_output_dirs(cfg)
    _validate_split(cfg)
    return cfg


def _validate_config(cfg: dict[str, Any]) -> None:
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in cfg:
            raise ValueError(f"Missing config section: {key}")
    for key in REQUIRED_PATH_KEYS:
        if key not in cfg["paths"]:
            raise ValueError(f"Missing paths.{key} in config")
    modalities = cfg["fusion"].get("modalities")
    if not isinstance(modalities, dict) or not modalities:
        raise ValueError("fusion.modalities must be a non-empty mapping")
    for name, modality_cfg in modalities.items():
        if "prediction_dir" not in modality_cfg:
            raise ValueError(f"Missing fusion.modalities.{name}.prediction_dir")


def _resolve_paths(cfg: dict[str, Any]) -> None:
    for key in REQUIRED_PATH_KEYS:
        p = Path(cfg["paths"][key])
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        cfg["paths"][key] = str(p)

    for modality_cfg in cfg["fusion"]["modalities"].values():
        p = Path(modality_cfg["prediction_dir"])
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        modality_cfg["prediction_dir"] = str(p)


def _validate_dirs(cfg: dict[str, Any]) -> None:
    for key in ["train_ann_dir", "val_ann_dir"]:
        p = Path(cfg["paths"][key])
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Input directory missing: paths.{key}={p}")
    for name, modality_cfg in cfg["fusion"]["modalities"].items():
        p = Path(modality_cfg["prediction_dir"])
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Input directory missing: fusion.modalities.{name}.prediction_dir={p}")


def _ensure_output_dirs(cfg: dict[str, Any]) -> None:
    Path(cfg["paths"]["prediction_dir"]).mkdir(parents=True, exist_ok=True)


def _validate_split(cfg: dict[str, Any]) -> None:
    split_cfg = cfg["split"]
    train_stories = set(split_cfg["stories_train"])
    val_stories = set(split_cfg["stories_val"])
    overlap = train_stories.intersection(val_stories)
    if overlap:
        raise ValueError(f"Train/val story overlap detected: {sorted(overlap)}")


def choose_device(device_flag: str):
    if torch is None:
        raise ImportError("PyTorch is required for late fusion alignment.")
    if device_flag == "cpu":
        return torch.device("cpu")
    if device_flag == "cuda":
        return torch.device("cuda")
    if device_flag == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ccc_numpy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mu_t = yt.mean()
    mu_p = yp.mean()
    var_t = yt.var()
    var_p = yp.var()
    cov = ((yt - mu_t) * (yp - mu_p)).mean()
    return float((2.0 * cov) / (var_t + var_p + (mu_p - mu_t) ** 2 + eps))


def iter_samples(cfg: dict[str, Any], split: str) -> list[SampleIndex]:
    split_cfg = cfg["split"]
    if split == "train":
        subjects = split_cfg["subjects_train"]
        stories = split_cfg["stories_train"]
    elif split == "val":
        subjects = split_cfg["subjects_val"]
        stories = split_cfg["stories_val"]
    else:
        raise ValueError(f"Invalid split: {split}")
    return [SampleIndex(subject=s, story=t, split=split) for s in subjects for t in stories]


def annotation_path(cfg: dict[str, Any], sample: SampleIndex) -> Path:
    base = cfg["paths"]["train_ann_dir"] if sample.split == "train" else cfg["paths"]["val_ann_dir"]
    return Path(base) / f"Subject_{sample.subject}_Story_{sample.story}.csv"


def read_labels(cfg: dict[str, Any], sample: SampleIndex) -> np.ndarray:
    path = annotation_path(cfg, sample)
    if not path.exists():
        raise FileNotFoundError(f"Annotation missing: {path}")
    y = pd.read_csv(path).iloc[:, 0].to_numpy(dtype=np.float32)
    return y.reshape(-1)


def modality_configs(cfg: dict[str, Any]) -> list[ModalityConfig]:
    modalities: list[ModalityConfig] = []
    for name, item in cfg["fusion"]["modalities"].items():
        modalities.append(
            ModalityConfig(
                name=name,
                prediction_dir=Path(str(item["prediction_dir"])),
                weight=float(item.get("weight", 1.0)),
                cutoff=float(item.get("cutoff", 0.0)),
                order=int(item.get("order", 1)),
                kind=str(item.get("kind", "frame")),
            )
        )
    return modalities


def butter_lowpass(cutoff: float, fs: float, order: int = 1) -> tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * fs
    if cutoff <= 0.0 or cutoff >= nyq:
        raise ValueError(f"Cutoff must be between 0 and Nyquist {nyq}, got {cutoff}")
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype="low", analog=False)


def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 1) -> np.ndarray:
    if len(data) == 0 or cutoff <= 0.0:
        return np.asarray(data, dtype=np.float32)
    b, a = butter_lowpass(cutoff, fs=fs, order=order)
    filtered = lfilter(b, a, np.asarray(data, dtype=np.float64))
    return filtered.astype(np.float32)


def butter_lowpass_filter_bidirectional(data: np.ndarray, cutoff: float, fs: float, order: int = 1) -> np.ndarray:
    x = np.asarray(data, dtype=np.float32).reshape(-1)
    if x.size == 0 or cutoff <= 0.0:
        return x.copy()
    first = butter_lowpass_filter(x[::-1], cutoff=cutoff, fs=fs, order=order)
    second = butter_lowpass_filter(first[::-1], cutoff=cutoff, fs=fs, order=order)
    return second.astype(np.float32)


def f_trick(y_train: np.ndarray, preds: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    y_flat = np.asarray(y_train, dtype=np.float32).reshape(-1)
    p_flat = np.asarray(preds, dtype=np.float32).reshape(-1)
    if y_flat.size == 0 or p_flat.size == 0:
        return p_flat.copy()

    s0 = float(np.std(y_flat))
    m0 = float(np.mean(y_flat))
    s1 = float(np.std(p_flat))
    m1 = float(np.mean(p_flat))
    if s1 <= eps:
        return np.full_like(p_flat, fill_value=m0, dtype=np.float32)
    norm_preds = s0 * (p_flat - m1) / s1 + m0
    return norm_preds.astype(np.float32)


def _prediction_file(modality: ModalityConfig, sample: SampleIndex) -> Path:
    return modality.prediction_dir / f"Subject_{sample.subject}_Story_{sample.story}.parquet"


def _validate_frame_prediction_df(df: pd.DataFrame) -> None:
    missing = [c for c in FRAME_MODALITY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required frame columns: {missing}")
    frame_idx = pd.Index(df["frame_idx"].to_numpy(dtype=np.int64))
    if frame_idx.duplicated().any() or not frame_idx.is_monotonic_increasing:
        raise ValueError("Invalid frame_idx: duplicates or not monotonic increasing")


def _validate_window_prediction_df(df: pd.DataFrame) -> None:
    missing = [c for c in WINDOW_MODALITY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required window columns: {missing}")
    window_idx = pd.Index(df["window_idx"].to_numpy(dtype=np.int64))
    if window_idx.duplicated().any() or not window_idx.is_monotonic_increasing:
        raise ValueError("Invalid window_idx: duplicates or not monotonic increasing")


def align_series_to_length(values: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    if target_len < 0:
        raise ValueError(f"target_len must be non-negative, got {target_len}")
    if target_len == 0:
        return np.zeros((0,), dtype=np.float32)
    if len(x) == target_len:
        return x
    if len(x) == 0:
        return np.zeros((target_len,), dtype=np.float32)
    if torch is None or F is None:
        raise ImportError("PyTorch is required to align prediction lengths.")
    tensor = torch.tensor(x, dtype=torch.float32).view(1, 1, -1)
    aligned = F.interpolate(tensor, size=target_len, mode="linear", align_corners=False)
    return aligned.view(-1).cpu().numpy().astype(np.float32)


def reconstruct_frame_series_from_windows(pred_df: pd.DataFrame, target_len: int) -> np.ndarray:
    _validate_window_prediction_df(pred_df)
    acc = np.zeros((target_len,), dtype=np.float32)
    cnt = np.zeros((target_len,), dtype=np.float32)
    starts = pred_df["window_start_frame"].to_numpy(dtype=np.int64)
    ends = pred_df["window_end_frame"].to_numpy(dtype=np.int64)
    preds = pred_df["y_pred"].to_numpy(dtype=np.float32)

    for start, end, pred in zip(starts, ends, preds):
        if not np.isfinite(pred):
            continue
        start_idx = max(0, int(start))
        end_idx = min(target_len - 1, int(end))
        if end_idx < start_idx:
            continue
        acc[start_idx : end_idx + 1] += float(pred)
        cnt[start_idx : end_idx + 1] += 1.0

    missing = cnt == 0.0
    if np.any(missing):
        valid_idx = np.flatnonzero(~missing)
        if valid_idx.size == 0:
            return np.zeros((target_len,), dtype=np.float32)
        acc[missing] = np.interp(np.flatnonzero(missing), valid_idx, acc[valid_idx] / cnt[valid_idx]).astype(np.float32)
        cnt[missing] = 1.0
    return (acc / cnt).astype(np.float32)


def read_modality_prediction(modality: ModalityConfig, sample: SampleIndex, target_len: int) -> np.ndarray:
    path = _prediction_file(modality, sample)
    if not path.exists():
        raise FileNotFoundError(f"Prediction missing for modality {modality.name}: {path}")
    df = pd.read_parquet(path)
    if modality.kind == "window":
        return reconstruct_frame_series_from_windows(df, target_len=target_len)
    if modality.kind != "frame":
        raise ValueError(f"Unsupported modality kind for {modality.name}: {modality.kind}")
    _validate_frame_prediction_df(df)
    values = df["y_pred"].to_numpy(dtype=np.float32)
    return align_series_to_length(values, target_len=target_len)


def read_training_labels_for_subject(cfg: dict[str, Any], subject_id: int) -> np.ndarray:
    train_labels = [read_labels(cfg, train_sample) for train_sample in iter_samples(cfg, "train") if train_sample.subject == subject_id]
    if not train_labels:
        raise FileNotFoundError(f"No training labels found for subject {subject_id}")
    return np.concatenate(train_labels, axis=0).astype(np.float32)


def build_modality_series_for_sample(cfg: dict[str, Any], sample: SampleIndex) -> dict[str, np.ndarray]:
    fps = float(cfg["fusion"].get("fps", 25.0))
    y_train = read_training_labels_for_subject(cfg, sample.subject)
    y_target = read_labels(cfg, sample)
    target_len = len(y_target)
    transformed: dict[str, np.ndarray] = {}

    for modality in modality_configs(cfg):
        preds = read_modality_prediction(modality, sample, target_len=target_len)
        preds = butter_lowpass_filter_bidirectional(preds, cutoff=modality.cutoff, fs=fps, order=modality.order)
        preds = f_trick(y_train, preds)
        transformed[modality.name] = preds.astype(np.float32)

    return transformed


def fuse_sample_predictions(cfg: dict[str, Any], sample: SampleIndex) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    final_cutoff = float(cfg["fusion"].get("final_cutoff", 0.0))
    final_order = int(cfg["fusion"].get("final_order", 1))
    fps = float(cfg["fusion"].get("fps", 25.0))
    y_train = read_training_labels_for_subject(cfg, sample.subject)
    transformed = build_modality_series_for_sample(cfg, sample)
    target_len = len(read_labels(cfg, sample))
    weighted_sum = np.zeros((target_len,), dtype=np.float32)
    total_weight = 0.0

    for modality in modality_configs(cfg):
        preds = transformed[modality.name] * modality.weight
        weighted_sum += preds
        total_weight += modality.weight

    if total_weight <= 0.0:
        raise ValueError("Total modality weight must be positive")

    fused = weighted_sum / total_weight
    fused = butter_lowpass_filter_bidirectional(fused, cutoff=final_cutoff, fs=fps, order=final_order)
    fused = f_trick(y_train, fused)
    return fused.astype(np.float32), transformed


def write_prediction_parquet(cfg: dict[str, Any], sample: SampleIndex, y_pred: np.ndarray) -> Path:
    out_dir = Path(cfg["paths"]["prediction_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = float(cfg["fusion"].get("fps", 25.0))
    frame_idx = np.arange(len(y_pred), dtype=np.int32)
    df = pd.DataFrame(
        {
            "frame_idx": frame_idx,
            "timestamp_s": frame_idx.astype(np.float32) / fps,
            "y_pred": y_pred.astype(np.float32),
            "subject_id": np.full(len(y_pred), sample.subject, dtype=np.int16),
            "story_id": np.full(len(y_pred), sample.story, dtype=np.int16),
            "split": [sample.split] * len(y_pred),
            "manifest_id": [cfg["split"]["manifest_id"]] * len(y_pred),
        }
    )
    out_path = out_dir / f"Subject_{sample.subject}_Story_{sample.story}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def validate_prediction_parquet(path: str | Path) -> None:
    df = pd.read_parquet(path)
    missing = [c for c in PREDICTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    if df["frame_idx"].duplicated().any() or not df["frame_idx"].is_monotonic_increasing:
        raise ValueError(f"Invalid frame_idx in {path}")
    if df["y_pred"].isna().any():
        raise ValueError(f"NaN y_pred values in {path}")


def now_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
