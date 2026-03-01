from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd
import yaml
try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    Dataset = object

REQUIRED_TOP_LEVEL_KEYS = ["paths", "split", "model", "train", "predict"]
REQUIRED_PATH_KEYS = [
    "srt_dir",
    "train_ann_dir",
    "val_ann_dir",
    "lexicon_dir",
    "feature_dir",
    "checkpoint_dir",
    "prediction_dir",
]


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML dictionary.")
    _validate_config(cfg)
    _resolve_paths(cfg, root=path.parent)
    _validate_dirs(cfg)
    _ensure_output_dirs(cfg)
    return cfg


def _validate_config(cfg: dict[str, Any]) -> None:
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in cfg:
            raise ValueError(f"Missing config section: {key}")
    for key in REQUIRED_PATH_KEYS:
        if key not in cfg["paths"]:
            raise ValueError(f"Missing paths.{key} in config")


def _resolve_paths(cfg: dict[str, Any], root: Path) -> None:
    del root  # kept for signature stability
    for key in REQUIRED_PATH_KEYS:
        p = Path(cfg["paths"][key])
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        cfg["paths"][key] = str(p)


def _validate_dirs(cfg: dict[str, Any]) -> None:
    for key in ["srt_dir", "train_ann_dir", "val_ann_dir", "lexicon_dir"]:
        p = Path(cfg["paths"][key])
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Input directory missing: paths.{key}={p}")


def _ensure_output_dirs(cfg: dict[str, Any]) -> None:
    for key in ["feature_dir", "checkpoint_dir", "prediction_dir"]:
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_flag: str) -> torch.device:
    if torch is None:
        raise ImportError("PyTorch is required for training/prediction.")
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


# Reusable, modality-agnostic CCC helpers.
def ccc_numpy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mu_t = yt.mean()
    mu_p = yp.mean()
    var_t = yt.var()
    var_p = yp.var()
    cov = ((yt - mu_t) * (yp - mu_p)).mean()
    return float((2.0 * cov) / (var_t + var_p + (mu_p - mu_t) ** 2 + eps))


def ccc_torch(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if torch is None:
        raise ImportError("PyTorch is required for ccc_torch.")
    yt = y_true.float().reshape(-1)
    yp = y_pred.float().reshape(-1)
    mu_t = torch.mean(yt)
    mu_p = torch.mean(yp)
    var_t = torch.var(yt, unbiased=False)
    var_p = torch.var(yp, unbiased=False)
    cov = torch.mean((yt - mu_t) * (yp - mu_p))
    return (2.0 * cov) / (var_t + var_p + (mu_p - mu_t) ** 2 + eps)


def ccc_loss_torch(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return 1.0 - ccc_torch(y_true, y_pred, eps=eps)


@dataclass
class SampleIndex:
    subject: int
    story: int
    split: str


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


def feature_path(cfg: dict[str, Any], sample: SampleIndex) -> Path:
    return Path(cfg["paths"]["feature_dir"]) / f"Subject_{sample.subject}_Story_{sample.story}_aligned.npy"


def read_labels(cfg: dict[str, Any], sample: SampleIndex) -> np.ndarray:
    ann = annotation_path(cfg, sample)
    if not ann.exists():
        raise FileNotFoundError(f"Annotation missing: {ann}")
    return pd.read_csv(ann).iloc[:, 0].to_numpy(dtype=np.float32)


def read_features(cfg: dict[str, Any], sample: SampleIndex) -> np.ndarray:
    feat = feature_path(cfg, sample)
    if not feat.exists():
        raise FileNotFoundError(f"Feature missing: {feat}")
    x = np.load(feat)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D features in {feat}, got shape {x.shape}")
    return x.astype(np.float32)


def window_sequence(x: np.ndarray, y: np.ndarray, window_size: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    if n < window_size:
        return np.empty((0, window_size, x.shape[-1]), dtype=np.float32), np.empty((0,), dtype=np.float32)

    starts = _window_starts(n, window_size=window_size, stride=stride)
    x_windows = []
    y_windows = []
    for start in starts:
        end = start + window_size
        x_windows.append(x[start:end])
        y_windows.append(float(np.mean(y[start:end])))
    return np.asarray(x_windows, dtype=np.float32), np.asarray(y_windows, dtype=np.float32)


def window_features(x: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    n = len(x)
    if n < window_size:
        return np.empty((0, window_size, x.shape[-1]), dtype=np.float32)
    starts = _window_starts(n, window_size=window_size, stride=stride)
    x_windows = [x[start : start + window_size] for start in starts]
    return np.asarray(x_windows, dtype=np.float32)


def _window_starts(n: int, window_size: int, stride: int) -> list[int]:
    return [i * stride for i in range(((n - window_size) // stride) + 1)]


class TranscriptWindowDataset(Dataset):
    def __init__(self, cfg: dict[str, Any], split: str, label_min: float | None = None, label_max: float | None = None):
        model_cfg = cfg["model"]
        window_size = int(model_cfg["window_size"])
        stride = int(model_cfg["stride"])

        rows: list[tuple[np.ndarray, int, float]] = []
        train_targets: list[float] = []

        for sample in iter_samples(cfg, split):
            try:
                x = read_features(cfg, sample)
                y = read_labels(cfg, sample)
            except FileNotFoundError:
                continue
            xw, yw = window_sequence(x, y, window_size=window_size, stride=stride)
            for xv, yv in zip(xw, yw):
                rows.append((xv, sample.subject - 1, float(yv)))
                train_targets.append(float(yv))

        self.label_min = float(np.min(train_targets)) if train_targets else -1.0
        self.label_max = float(np.max(train_targets)) if train_targets else 1.0
        if label_min is not None:
            self.label_min = float(label_min)
        if label_max is not None:
            self.label_max = float(label_max)

        self.rows: list[tuple[np.ndarray, int, float]] = []
        denom = self.label_max - self.label_min + 1e-8
        for xv, sid, yv in rows:
            self.rows.append((xv, sid, (yv - self.label_min) / denom))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        if torch is None:
            raise ImportError("PyTorch is required for TranscriptWindowDataset.")
        x, subject_idx, y = self.rows[idx]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(subject_idx, dtype=torch.long),
            torch.tensor(y, dtype=torch.float32),
        )


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class TranscriptLSTMModel(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        if torch is None or nn is None:
            raise ImportError("PyTorch is required for TranscriptLSTMModel.")
        super().__init__()
        m = cfg["model"]
        embedding_size = int(m["embedding_size"])
        lstm_hidden_dim = int(m["lstm_hidden_dim"])
        subject_embed_dim = int(m["subject_embed_dim"])
        dense_hidden_dim = int(m["dense_hidden_dim"])
        dropout = float(m["dropout"])

        all_subjects = set(cfg["split"]["subjects_train"]) | set(cfg["split"]["subjects_val"])
        subject_count = max(all_subjects)

        self.lstm = nn.LSTM(embedding_size, lstm_hidden_dim, batch_first=True)
        self.attn = AttentionPooling(lstm_hidden_dim)
        self.subject_embed = nn.Embedding(subject_count, subject_embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim + subject_embed_dim, dense_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, subject_idx: torch.Tensor) -> torch.Tensor:
        seq, _ = self.lstm(x)
        pooled = self.attn(seq)
        sid = self.subject_embed(subject_idx)
        return self.head(torch.cat([pooled, sid], dim=1)).squeeze(-1)


# def write_prediction_parquet(cfg: dict[str, Any], sample: SampleIndex, y_pred: np.ndarray) -> Path:
#     out_dir = Path(cfg["paths"]["prediction_dir"])
#     out_dir.mkdir(parents=True, exist_ok=True)
#     frame_idx = np.arange(len(y_pred), dtype=np.int32)
#     df = pd.DataFrame(
#         {
#             "frame_idx": frame_idx,
#             "timestamp_s": frame_idx.astype(np.float32) / 25.0,
#             "y_pred": y_pred.astype(np.float32),
#             "subject_id": np.full(len(y_pred), sample.subject, dtype=np.int16),
#             "story_id": np.full(len(y_pred), sample.story, dtype=np.int16),
#             "split": [sample.split] * len(y_pred),
#             "manifest_id": [cfg["split"]["manifest_id"]] * len(y_pred),
#         }
#     )
#     out_path = out_dir / f"Subject_{sample.subject}_Story_{sample.story}.parquet"
#     df.to_parquet(out_path, index=False)
#     return out_path
def write_prediction_parquet(cfg: dict[str, Any], sample: SampleIndex, y_pred: np.ndarray) -> Path:
    out_dir = Path(cfg["paths"]["prediction_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    window_size = int(cfg["model"]["window_size"])
    stride = int(cfg["model"]["stride"])
    window_idx = np.arange(len(y_pred), dtype=np.int32)
    window_start_frame = (window_idx * stride).astype(np.int32)
    window_end_frame = (window_start_frame + window_size - 1).astype(np.int32)
    window_center_frame = (window_start_frame + (window_size // 2)).astype(np.int32)
    df = pd.DataFrame(
        {
            "window_idx": window_idx,
            "window_start_frame": window_start_frame,
            "window_end_frame": window_end_frame,
            "window_center_frame": window_center_frame,
            "window_center_s": window_center_frame.astype(np.float32) / 25.0,
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


# def validate_prediction_parquet(path: str | Path) -> None:
#     required = ["frame_idx", "timestamp_s", "y_pred", "subject_id", "story_id", "split", "manifest_id"]
#     df = pd.read_parquet(path)
#     missing = [c for c in required if c not in df.columns]
#     if missing:
#         raise ValueError(f"Missing columns in {path}: {missing}")
#     if df["frame_idx"].duplicated().any() or not df["frame_idx"].is_monotonic_increasing:
#         raise ValueError(f"Invalid frame_idx in {path}")
#     if df["y_pred"].isna().any():
#         raise ValueError(f"NaN y_pred values in {path}")
def validate_prediction_parquet(path: str | Path) -> None:
    required = [
        "window_idx",
        "window_start_frame",
        "window_end_frame",
        "window_center_frame",
        "window_center_s",
        "y_pred",
        "subject_id",
        "story_id",
        "split",
        "manifest_id",
    ]
    df = pd.read_parquet(path)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    if df["window_idx"].duplicated().any() or not df["window_idx"].is_monotonic_increasing:
        raise ValueError(f"Invalid window_idx in {path}")
    if df["y_pred"].isna().any():
        raise ValueError(f"NaN y_pred values in {path}")


def checkpoint_path(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["checkpoint_dir"]) / "transcript_lstm.pt"


def now_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
