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


REQUIRED_TOP_LEVEL_KEYS = ["paths", "split", "audio", "extract", "model", "train", "predict"]
REQUIRED_PATH_KEYS = [
    "train_videos_dir",
    "val_videos_dir",
    "predictor_path",
    "train_ann_dir",
    "val_ann_dir",
    "train_landmarks_csv_dir",
    "val_landmarks_csv_dir",
    "feature_dir",
    "checkpoint_dir",
    "prediction_dir",
]

PREDICTION_COLUMNS = [
    "frame_idx",
    "timestamp_s",
    "y_pred",
    "subject_id",
    "story_id",
    "split",
    "manifest_id",
]


@dataclass
class SampleIndex:
    subject: int
    story: int
    split: str


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
    _validate_split(cfg)
    return cfg


def _validate_config(cfg: dict[str, Any]) -> None:
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in cfg:
            raise ValueError(f"Missing config section: {key}")
    for key in REQUIRED_PATH_KEYS:
        if key not in cfg["paths"]:
            raise ValueError(f"Missing paths.{key} in config")


def _resolve_paths(cfg: dict[str, Any], root: Path) -> None:
    del root
    for key in REQUIRED_PATH_KEYS:
        p = Path(cfg["paths"][key])
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        cfg["paths"][key] = str(p)


def _validate_dirs(cfg: dict[str, Any]) -> None:
    for key in ["train_videos_dir", "val_videos_dir", "train_ann_dir", "val_ann_dir"]:
        p = Path(cfg["paths"][key])
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Input directory missing: paths.{key}={p}")
    predictor = Path(cfg["paths"]["predictor_path"])
    if not predictor.exists() or not predictor.is_file():
        raise FileNotFoundError(f"Predictor file missing: paths.predictor_path={predictor}")


def _ensure_output_dirs(cfg: dict[str, Any]) -> None:
    for key in ["train_landmarks_csv_dir", "val_landmarks_csv_dir", "feature_dir", "checkpoint_dir", "prediction_dir"]:
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


def _validate_split(cfg: dict[str, Any]) -> None:
    split_cfg = cfg["split"]
    overlap = set(split_cfg["stories_train"]).intersection(split_cfg["stories_val"])
    if overlap:
        raise ValueError(f"Train/val story overlap detected: {sorted(overlap)}")


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


def video_path(cfg: dict[str, Any], sample: SampleIndex) -> Path:
    base = cfg["paths"]["train_videos_dir"] if sample.split == "train" else cfg["paths"]["val_videos_dir"]
    return Path(base) / f"Subject_{sample.subject}_Story_{sample.story}.mp4"


def landmark_csv_path(cfg: dict[str, Any], sample: SampleIndex) -> Path:
    base = cfg["paths"]["train_landmarks_csv_dir"] if sample.split == "train" else cfg["paths"]["val_landmarks_csv_dir"]
    return Path(base) / f"Subject_{sample.subject}_Story_{sample.story}" / "Subject_face_landmarks.csv"


def feature_path(cfg: dict[str, Any], sample: SampleIndex) -> Path:
    return Path(cfg["paths"]["feature_dir"]) / "aligned" / f"Subject_{sample.subject}_Story_{sample.story}_aligned.npz"


def read_labels(cfg: dict[str, Any], sample: SampleIndex) -> np.ndarray:
    ann = annotation_path(cfg, sample)
    if not ann.exists():
        raise FileNotFoundError(f"Annotation missing: {ann}")
    y = pd.read_csv(ann).iloc[:, 0].to_numpy(dtype=np.float32)
    return y.reshape(-1)


def read_landmarks(cfg: dict[str, Any], sample: SampleIndex) -> np.ndarray:
    path = landmark_csv_path(cfg, sample)
    if not path.exists():
        raise FileNotFoundError(f"Landmark CSV missing: {path}")
    x = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 136:
        raise ValueError(f"Expected 136 features for landmarks, got {x.shape}")
    return x


def load_aligned(cfg: dict[str, Any], sample: SampleIndex) -> tuple[np.ndarray, np.ndarray]:
    path = feature_path(cfg, sample)
    if not path.exists():
        raise FileNotFoundError(f"Aligned feature missing: {path}")
    d = np.load(path)
    return d["x"].astype(np.float32), d["y"].astype(np.float32)


def build_aligned(cfg: dict[str, Any], sample: SampleIndex) -> Path:
    x = read_landmarks(cfg, sample)
    y = read_labels(cfg, sample)
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    out = feature_path(cfg, sample)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x=x.astype(np.float32), y=y.astype(np.float32))
    return out


def window_landmarks(x: np.ndarray, window_size: int) -> np.ndarray:
    n, d = x.shape
    out = np.zeros((n, window_size, d), dtype=np.float32)
    pad_idx = window_size - 1
    if n == 0:
        return out

    for i in range(min(pad_idx, n)):
        out[i] = x[min(pad_idx, n - 1)]
    for i in range(pad_idx, n):
        out[i] = x[i - pad_idx : i + 1]
    if n < window_size:
        out[:] = x[-1]
    return out


class LandmarksWindowDataset(Dataset):
    def __init__(
        self,
        cfg: dict[str, Any],
        split: str,
        feature_mean: float | None = None,
        feature_std: float | None = None,
        target_min: float | None = None,
        target_max: float | None = None,
    ):
        if torch is None:
            raise ImportError("PyTorch is required for LandmarksWindowDataset.")

        window_size = int(cfg["model"]["window_size"])
        x_rows = []
        y_rows = []

        for sample in iter_samples(cfg, split):
            try:
                x, y = load_aligned(cfg, sample)
            except FileNotFoundError:
                continue
            n = min(len(x), len(y))
            if n == 0:
                continue
            xw = window_landmarks(x[:n], window_size)
            x_rows.append(xw)
            y_rows.append(y[:n].reshape(-1, 1))

        if x_rows:
            x_np = np.concatenate(x_rows, axis=0)
            y_np = np.concatenate(y_rows, axis=0)
        else:
            x_np = np.empty((0, window_size, 136), dtype=np.float32)
            y_np = np.empty((0, 1), dtype=np.float32)

        self.feature_mean = float(np.mean(x_np)) if feature_mean is None and x_np.size else 0.0
        self.feature_std = float(np.std(x_np)) if feature_std is None and x_np.size else 1.0
        if feature_mean is not None:
            self.feature_mean = float(feature_mean)
        if feature_std is not None:
            self.feature_std = float(feature_std)
        if self.feature_std < 1e-8:
            self.feature_std = 1.0

        self.target_min = float(np.min(y_np)) if target_min is None and y_np.size else -1.0
        self.target_max = float(np.max(y_np)) if target_max is None and y_np.size else 1.0
        if target_min is not None:
            self.target_min = float(target_min)
        if target_max is not None:
            self.target_max = float(target_max)

        y_scale = self.target_max - self.target_min
        if abs(y_scale) < 1e-8:
            y_scale = 1.0

        self.x = ((x_np - self.feature_mean) / self.feature_std).astype(np.float32)
        self.y = ((y_np - self.target_min) / y_scale).astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.x[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


class LandmarksConv1DModel(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        if nn is None:
            raise ImportError("PyTorch is required for LandmarksConv1DModel.")
        super().__init__()
        m = cfg["model"]
        channels = list(m["conv_channels"])
        k = int(m["kernel_size"])
        dense_dim = int(m["dense_dim"])
        dropout = float(m["dropout"])

        self.backbone = nn.Sequential(
            nn.Conv1d(136, int(channels[0]), k),
            nn.BatchNorm1d(int(channels[0])),
            nn.ReLU(),
            nn.Conv1d(int(channels[0]), int(channels[1]), k),
            nn.ReLU(),
            nn.Conv1d(int(channels[1]), int(channels[2]), k),
            nn.ReLU(),
            nn.Conv1d(int(channels[2]), int(channels[3]), k),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(int(channels[3]), dense_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.backbone(x)
        return self.head(x)


def denorm_target(y_norm: np.ndarray, target_min: float, target_max: float) -> np.ndarray:
    return y_norm * (target_max - target_min + 1e-8) + target_min


def write_prediction_parquet(cfg: dict[str, Any], sample: SampleIndex, y_pred: np.ndarray) -> Path:
    out_dir = Path(cfg["paths"]["prediction_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = float(cfg["audio"]["fps"])
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


def checkpoint_path(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["checkpoint_dir"]) / "landmarks_conv1d.pt"


def now_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
