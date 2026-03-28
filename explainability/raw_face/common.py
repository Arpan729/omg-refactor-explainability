from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import random
import re
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml

try:
    import dlib
except Exception:  # pragma: no cover
    dlib = None

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
    "train_ann_dir",
    "val_ann_dir",
    "train_faces_dir",
    "val_faces_dir",
    "feature_dir",
    "checkpoint_dir",
    "prediction_dir",
]
PREDICTION_COLUMNS = ["frame_idx", "timestamp_s", "y_pred", "subject_id", "story_id", "split", "manifest_id"]


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


def _resolve_paths(cfg: dict[str, Any]) -> None:
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


def _ensure_output_dirs(cfg: dict[str, Any]) -> None:
    for key in ["train_faces_dir", "val_faces_dir", "feature_dir", "checkpoint_dir", "prediction_dir"]:
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


def _validate_split(cfg: dict[str, Any]) -> None:
    overlap = set(cfg["split"]["stories_train"]).intersection(cfg["split"]["stories_val"])
    if overlap:
        raise ValueError(f"Train/val story overlap detected: {sorted(overlap)}")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def choose_device(device_flag: str) -> torch.device:
    if torch is None:
        raise ImportError("PyTorch is required.")
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
    if split == "train":
        subjects = cfg["split"]["subjects_train"]
        stories = cfg["split"]["stories_train"]
    elif split == "val":
        subjects = cfg["split"]["subjects_val"]
        stories = cfg["split"]["stories_val"]
    else:
        raise ValueError(f"Invalid split: {split}")
    return [SampleIndex(subject=s, story=t, split=split) for s in subjects for t in stories]


def video_path(cfg: dict[str, Any], sample: SampleIndex) -> Path:
    base = cfg["paths"]["train_videos_dir"] if sample.split == "train" else cfg["paths"]["val_videos_dir"]
    return Path(base) / f"Subject_{sample.subject}_Story_{sample.story}.mp4"


def annotation_path(cfg: dict[str, Any], sample: SampleIndex) -> Path:
    base = cfg["paths"]["train_ann_dir"] if sample.split == "train" else cfg["paths"]["val_ann_dir"]
    return Path(base) / f"Subject_{sample.subject}_Story_{sample.story}.csv"


def face_sample_dir(cfg: dict[str, Any], sample: SampleIndex) -> Path:
    base = cfg["paths"]["train_faces_dir"] if sample.split == "train" else cfg["paths"]["val_faces_dir"]
    return Path(base) / f"Subject_{sample.subject}_Story_{sample.story}" / "Subject_face"


def feature_path(cfg: dict[str, Any], sample: SampleIndex) -> Path:
    return Path(cfg["paths"]["feature_dir"]) / f"Subject_{sample.subject}_Story_{sample.story}_aligned.npz"


def checkpoint_path(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["checkpoint_dir"]) / "raw_face_3dcnn.pt"


def now_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def read_labels(cfg: dict[str, Any], sample: SampleIndex) -> np.ndarray:
    path = annotation_path(cfg, sample)
    if not path.exists():
        raise FileNotFoundError(f"Annotation missing: {path}")
    y = pd.read_csv(path).iloc[:, 0].to_numpy(dtype=np.float32)
    return y.reshape(-1)


def sorted_face_frames(face_dir: Path, ext: str = ".png") -> list[Path]:
    frames = [p for p in face_dir.iterdir() if p.is_file() and p.suffix.lower() == ext.lower()]
    def _key(p: Path):
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else 10**9
    return sorted(frames, key=_key)


def extract_subject_faces_for_sample(cfg: dict[str, Any], sample: SampleIndex, detector) -> bool:
    vp = video_path(cfg, sample)
    if not vp.exists():
        return False

    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        return False

    out_dir = face_sample_dir(cfg, sample)
    out_dir.mkdir(parents=True, exist_ok=True)

    upsample = int(cfg["extract"]["detector_upsample"])
    size = int(cfg["extract"]["face_size"])
    ext = str(cfg["extract"]["image_ext"])

    prev_box = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        h, w = frame.shape[:2]
        mid = w // 2
        subject_half = frame[:, mid:]

        dets = detector(subject_half, upsample)
        best = None
        if dets:
            best = max(dets, key=lambda d: max(0, d.right() - d.left()) * max(0, d.bottom() - d.top()))

        if best is not None:
            x1, y1, x2, y2 = best.left(), best.top(), best.right(), best.bottom()
            x1 = max(0, min(x1, subject_half.shape[1] - 1))
            y1 = max(0, min(y1, subject_half.shape[0] - 1))
            x2 = max(x1 + 1, min(x2, subject_half.shape[1]))
            y2 = max(y1 + 1, min(y2, subject_half.shape[0]))
            prev_box = (x1, y1, x2, y2)
        elif prev_box is not None:
            x1, y1, x2, y2 = prev_box
        else:
            x1, y1, x2, y2 = 0, 0, subject_half.shape[1], subject_half.shape[0]

        crop = subject_half[y1:y2, x1:x2]
        if crop.size == 0:
            crop = subject_half

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"{idx}{ext}"), resized)
        idx += 1

    cap.release()
    return idx > 0


def read_face_tensor(cfg: dict[str, Any], sample: SampleIndex) -> np.ndarray:
    face_dir = face_sample_dir(cfg, sample)
    if not face_dir.exists():
        raise FileNotFoundError(f"Face directory missing: {face_dir}")
    ext = str(cfg["extract"]["image_ext"])
    paths = sorted_face_frames(face_dir, ext=ext)
    if not paths:
        raise FileNotFoundError(f"No face frames found: {face_dir}")

    imgs = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = img.astype(np.float32) / 255.0
        m = float(np.mean(img))
        s = float(np.std(img))
        img = (img - m) if s < 1e-8 else (img - m) / s
        imgs.append(img)
    if not imgs:
        raise ValueError(f"No decodable images in: {face_dir}")

    x = np.asarray(imgs, dtype=np.float32)
    x = x[:, np.newaxis, :, :]
    return x


def build_aligned(cfg: dict[str, Any], sample: SampleIndex) -> Path:
    x = read_face_tensor(cfg, sample)
    y = read_labels(cfg, sample)
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    sid = np.full((n,), sample.subject - 1, dtype=np.int64)
    out = feature_path(cfg, sample)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x=x.astype(np.float32), y=y.astype(np.float32), sid=sid)
    return out


def load_aligned(cfg: dict[str, Any], sample: SampleIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = feature_path(cfg, sample)
    if not path.exists():
        raise FileNotFoundError(f"Aligned feature missing: {path}")
    d = np.load(path)
    return d["x"].astype(np.float32), d["y"].astype(np.float32), d["sid"].astype(np.int64)


def window_sequence(x: np.ndarray, y: np.ndarray, sid: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = min(len(x), len(y), len(sid))
    x = x[:n]
    y = y[:n]
    sid = sid[:n]
    if n == 0:
        sz = x.shape[-1] if x.ndim == 4 else 48
        return (
            np.empty((0, 1, seq_len, sz, sz), dtype=np.float32),
            np.empty((0, 1), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    w = []
    t = []
    s = []
    for i in range(n):
        start = max(0, i - seq_len + 1)
        chunk = x[start : i + 1]
        if len(chunk) < seq_len:
            pad = np.repeat(chunk[:1], seq_len - len(chunk), axis=0)
            chunk = np.concatenate([pad, chunk], axis=0)
        chunk = np.transpose(chunk, (1, 0, 2, 3))
        w.append(chunk)
        t.append(y[i])
        s.append(sid[i])
    return np.asarray(w, dtype=np.float32), np.asarray(t, dtype=np.float32).reshape(-1, 1), np.asarray(s, dtype=np.int64)


class RawFaceWindowDataset(Dataset):
    def __init__(
        self,
        cfg: dict[str, Any],
        split: str,
        target_min: float | None = None,
        target_max: float | None = None,
    ):
        if torch is None:
            raise ImportError("PyTorch is required for RawFaceWindowDataset.")

        seq_len = int(cfg["model"]["seq_len"])
        x_rows = []
        y_rows = []
        sid_rows = []

        for sample in iter_samples(cfg, split):
            try:
                x, y, sid = load_aligned(cfg, sample)
            except FileNotFoundError:
                continue
            xw, yw, sw = window_sequence(x, y, sid, seq_len=seq_len)
            x_rows.append(xw)
            y_rows.append(yw)
            sid_rows.append(sw)

        if x_rows:
            self.x = np.concatenate(x_rows, axis=0)
            y_np = np.concatenate(y_rows, axis=0).reshape(-1)
            self.sid = np.concatenate(sid_rows, axis=0)
        else:
            size = int(cfg["extract"]["face_size"])
            self.x = np.empty((0, 1, seq_len, size, size), dtype=np.float32)
            y_np = np.empty((0,), dtype=np.float32)
            self.sid = np.empty((0,), dtype=np.int64)

        self.target_min = float(np.min(y_np)) if target_min is None and y_np.size else -1.0
        self.target_max = float(np.max(y_np)) if target_max is None and y_np.size else 1.0
        if target_min is not None:
            self.target_min = float(target_min)
        if target_max is not None:
            self.target_max = float(target_max)

        scale = self.target_max - self.target_min
        if abs(scale) < 1e-8:
            scale = 1.0
        self.y = ((y_np - self.target_min) / scale).astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.x[idx], dtype=torch.float32),
            torch.tensor(self.sid[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


class RawFace3DCNNModel(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        if nn is None:
            raise ImportError("PyTorch is required for RawFace3DCNNModel.")
        super().__init__()
        m = cfg["model"]
        c1, c2 = [int(v) for v in m["conv_channels"]]
        dense_dim = int(m["dense_dim"])
        embed_dim = int(m["subject_embed_dim"])
        dropout = float(m["dropout"])

        subject_count = max(set(cfg["split"]["subjects_train"]) | set(cfg["split"]["subjects_val"]))

        self.backbone = nn.Sequential(
            nn.Conv3d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(c1, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(c1),
            nn.Conv3d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(c2, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(c2),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        self.img_fc = nn.Linear(c2, dense_dim)
        self.sid_embed = nn.Embedding(subject_count, embed_dim)
        self.sid_fc = nn.Linear(embed_dim, 3)

        self.head = nn.Sequential(
            nn.Linear(dense_dim + 3, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, sid: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x).flatten(1)
        h = torch.relu(self.img_fc(h))
        s = torch.relu(self.sid_fc(self.sid_embed(sid)))
        return self.head(torch.cat([h, s], dim=1))


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
