from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import re
from typing import Any

import cv2
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
    "train_ann_dir",
    "val_ann_dir",
    "train_fullbody_dir",
    "val_fullbody_dir",
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
    for key in ["train_fullbody_dir", "val_fullbody_dir", "feature_dir", "checkpoint_dir", "prediction_dir"]:
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


def subject_image_dir(cfg: dict[str, Any], sample: SampleIndex) -> Path:
    base = cfg["paths"]["train_fullbody_dir"] if sample.split == "train" else cfg["paths"]["val_fullbody_dir"]
    return Path(base) / f"Subject_{sample.subject}_Story_{sample.story}" / "Subject_img"


def actor_image_dir(cfg: dict[str, Any], sample: SampleIndex) -> Path:
    base = cfg["paths"]["train_fullbody_dir"] if sample.split == "train" else cfg["paths"]["val_fullbody_dir"]
    return Path(base) / f"Subject_{sample.subject}_Story_{sample.story}" / "Actor_img"


def feature_path(cfg: dict[str, Any], sample: SampleIndex) -> Path:
    return Path(cfg["paths"]["feature_dir"]) / f"Subject_{sample.subject}_Story_{sample.story}_aligned.npz"


def checkpoint_path(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["checkpoint_dir"]) / "fullbody_resnet3d.pt"


def _define_frames(tag: str, size: int = 620, x_shift: int = 0, y_shift: int = 0) -> tuple[int, int, int, int]:
    if tag == "actor":
        start_x = 290 + x_shift
    elif tag == "subject":
        start_x = 1460 + x_shift
    else:
        raise ValueError("Specify tag as actor or subject")

    start_y = 720 - size + y_shift
    end_x = start_x + size
    end_y = start_y + size
    return start_x, start_y, end_x, end_y


def _coords_for_video(video_stem: str) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    if video_stem == "Subject_2_Story_8":
        actor_box = _define_frames(x_shift=-20, tag="actor")
    elif video_stem == "Subject_4_Story_4":
        actor_box = _define_frames(x_shift=-20, tag="actor")
    elif video_stem == "Subject_4_Story_5":
        actor_box = _define_frames(x_shift=80, tag="actor")
    else:
        actor_box = _define_frames(tag="actor")

    if video_stem == "Subject_1_Story_5":
        subject_box = _define_frames(x_shift=-50, tag="subject")
    elif video_stem == "Subject_2_Story_8":
        subject_box = _define_frames(x_shift=-20, tag="subject")
    else:
        subject_box = _define_frames(tag="subject")

    return actor_box, subject_box


def extract_fullbody_for_sample(cfg: dict[str, Any], sample: SampleIndex, preview_frames: int | None = None) -> int:
    vp = video_path(cfg, sample)
    if not vp.exists():
        return 0

    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        return 0

    size = int(cfg["extract"]["image_size"])
    ext = str(cfg["extract"]["image_ext"])

    actor_dir = actor_image_dir(cfg, sample)
    subject_dir = subject_image_dir(cfg, sample)
    actor_dir.mkdir(parents=True, exist_ok=True)
    subject_dir.mkdir(parents=True, exist_ok=True)

    actor_box, subject_box = _coords_for_video(vp.stem)

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        x1, y1, x2, y2 = actor_box
        actor = frame[y1:y2, x1:x2]
        x1, y1, x2, y2 = subject_box
        subject = frame[y1:y2, x1:x2]

        if actor.size == 0:
            actor = frame
        if subject.size == 0:
            subject = frame

        actor_gray = cv2.cvtColor(actor, cv2.COLOR_BGR2GRAY) if actor.ndim == 3 else actor
        subject_gray = cv2.cvtColor(subject, cv2.COLOR_BGR2GRAY) if subject.ndim == 3 else subject

        actor_resized = cv2.resize(actor_gray, (size, size), interpolation=cv2.INTER_AREA)
        subject_resized = cv2.resize(subject_gray, (size, size), interpolation=cv2.INTER_AREA)

        cv2.imwrite(str(actor_dir / f"{idx}{ext}"), actor_resized)
        cv2.imwrite(str(subject_dir / f"{idx}{ext}"), subject_resized)

        idx += 1
        if preview_frames is not None and idx >= preview_frames:
            break

    cap.release()
    return idx


def _sorted_image_frames(img_dir: Path, ext: str) -> list[Path]:
    paths = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() == ext.lower()]

    def _key(p: Path):
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else 10**9

    return sorted(paths, key=_key)


def read_subject_tensor(cfg: dict[str, Any], sample: SampleIndex) -> np.ndarray:
    img_dir = subject_image_dir(cfg, sample)
    if not img_dir.exists():
        raise FileNotFoundError(f"Subject image directory missing: {img_dir}")

    ext = str(cfg["extract"]["image_ext"])
    paths = _sorted_image_frames(img_dir, ext=ext)
    if not paths:
        raise FileNotFoundError(f"No subject images found: {img_dir}")

    step = max(1, int(cfg["model"]["down_sampling"]))
    paths = paths[::step]

    imgs: list[np.ndarray] = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = img.astype(np.float32)
        mean = float(np.mean(img))
        std = float(np.std(img))
        img = (img - mean) if std < 1e-8 else (img - mean) / std
        imgs.append(img)

    if not imgs:
        raise ValueError(f"No decodable images found in {img_dir}")

    x = np.asarray(imgs, dtype=np.float32)
    x = x[:, np.newaxis, :, :]
    return x


def read_labels(cfg: dict[str, Any], sample: SampleIndex) -> np.ndarray:
    path = annotation_path(cfg, sample)
    if not path.exists():
        raise FileNotFoundError(f"Annotation missing: {path}")

    try:
        y = np.loadtxt(path, dtype=np.float32, skiprows=1)
    except Exception:
        y = pd.read_csv(path).iloc[:, 0].to_numpy(dtype=np.float32)

    if y.ndim > 1:
        y = y.reshape(-1)

    step = max(1, int(cfg["model"]["down_sampling"]))
    return y[::step].reshape(-1)


def build_aligned(cfg: dict[str, Any], sample: SampleIndex) -> Path:
    x = read_subject_tensor(cfg, sample)
    y = read_labels(cfg, sample)
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    out = feature_path(cfg, sample)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x=x.astype(np.float32), y=y.astype(np.float32))
    return out


def load_aligned(cfg: dict[str, Any], sample: SampleIndex) -> tuple[np.ndarray, np.ndarray]:
    path = feature_path(cfg, sample)
    if not path.exists():
        raise FileNotFoundError(f"Aligned feature missing: {path}")
    d = np.load(path)
    return d["x"].astype(np.float32), d["y"].astype(np.float32)


def window_sequence_legacy(x: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    if n <= seq_len:
        h = x.shape[-2] if x.ndim == 4 else int(48)
        w = x.shape[-1] if x.ndim == 4 else int(48)
        return (
            np.empty((0, 1, seq_len, h, w), dtype=np.float32),
            np.empty((0, 1), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    windows = []
    targets = []
    frame_idx = []
    for i in range(n - seq_len):
        chunk = x[i : i + seq_len]
        chunk = np.transpose(chunk, (1, 0, 2, 3))
        windows.append(chunk)
        targets.append(y[i + seq_len])
        frame_idx.append(i + seq_len)

    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(targets, dtype=np.float32).reshape(-1, 1),
        np.asarray(frame_idx, dtype=np.int64),
    )


class FullBodyWindowDataset(Dataset):
    def __init__(
        self,
        cfg: dict[str, Any],
        split: str,
        target_min: float | None = None,
        target_max: float | None = None,
    ):
        if torch is None:
            raise ImportError("PyTorch is required for FullBodyWindowDataset.")

        seq_len = int(cfg["model"]["seq_len"])
        x_rows = []
        y_rows = []

        for sample in iter_samples(cfg, split):
            try:
                x, y = load_aligned(cfg, sample)
            except FileNotFoundError:
                continue
            xw, yw, _ = window_sequence_legacy(x, y, seq_len=seq_len)
            x_rows.append(xw)
            y_rows.append(yw)

        size = int(cfg["extract"]["image_size"])
        if x_rows:
            self.x = np.concatenate(x_rows, axis=0)
            y_np = np.concatenate(y_rows, axis=0).reshape(-1)
        else:
            self.x = np.empty((0, 1, seq_len, size, size), dtype=np.float32)
            y_np = np.empty((0,), dtype=np.float32)

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
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out


class FullBodyResNet3DModel(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        if nn is None:
            raise ImportError("PyTorch is required for FullBodyResNet3DModel.")
        super().__init__()

        hidden_dim = int(cfg["model"]["hidden_dim"])

        self.inplanes = 64
        self.stem = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc1 = nn.Linear(512, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock3D(self.inplanes, planes, stride=stride)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(self.inplanes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def denorm_target(y_norm: np.ndarray, target_min: float, target_max: float) -> np.ndarray:
    return y_norm * (target_max - target_min + 1e-8) + target_min


def write_prediction_parquet(
    cfg: dict[str, Any],
    sample: SampleIndex,
    y_pred: np.ndarray,
    frame_idx: np.ndarray | None = None,
) -> Path:
    out_dir = Path(cfg["paths"]["prediction_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if frame_idx is None:
        frame_idx_np = np.arange(len(y_pred), dtype=np.int32)
    else:
        frame_idx_np = np.asarray(frame_idx, dtype=np.int32).reshape(-1)
        if len(frame_idx_np) != len(y_pred):
            raise ValueError("frame_idx length must match y_pred length")

    fps = float(cfg["audio"]["fps"])
    df = pd.DataFrame(
        {
            "frame_idx": frame_idx_np,
            "timestamp_s": frame_idx_np.astype(np.float32) / fps,
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
