from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import random
import wave
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None
    F = None
    nn = None
    Dataset = object


REQUIRED_TOP_LEVEL_KEYS = [
    "paths",
    "split",
    "audio",
    "feature",
    "model",
    "train",
    "predict",
]
REQUIRED_PATH_KEYS = [
    "train_ann_dir",
    "val_ann_dir",
    "train_audio_dir",
    "val_audio_dir",
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
    del root  # kept for signature stability
    for key in REQUIRED_PATH_KEYS:
        p = Path(cfg["paths"][key])
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        cfg["paths"][key] = str(p)


def _validate_dirs(cfg: dict[str, Any]) -> None:
    for key in ["train_ann_dir", "val_ann_dir", "train_audio_dir", "val_audio_dir"]:
        p = Path(cfg["paths"][key])
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Input directory missing: paths.{key}={p}")


def _ensure_output_dirs(cfg: dict[str, Any]) -> None:
    for key in ["feature_dir", "checkpoint_dir", "prediction_dir"]:
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


def _validate_split(cfg: dict[str, Any]) -> None:
    split_cfg = cfg["split"]
    train_stories = set(split_cfg["stories_train"])
    val_stories = set(split_cfg["stories_val"])
    overlap = train_stories.intersection(val_stories)
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


def ccc_torch_sequence(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if torch is None:
        raise ImportError("PyTorch is required for ccc_torch_sequence.")
    yt = y_true.float()
    yp = y_pred.float()
    mu_t = torch.mean(yt, dim=1)
    mu_p = torch.mean(yp, dim=1)
    ct = yt - mu_t.unsqueeze(1)
    cp = yp - mu_p.unsqueeze(1)
    cov = torch.mean(ct * cp, dim=1)
    var_t = torch.mean(ct * ct, dim=1)
    var_p = torch.mean(cp * cp, dim=1)
    ccc = (2.0 * cov) / (var_t + var_p + (mu_p - mu_t) ** 2 + eps)
    return torch.mean(ccc)


def ccc_loss_torch_sequence(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return 1.0 - ccc_torch_sequence(y_true, y_pred, eps=eps)


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


def audio_path(cfg: dict[str, Any], sample: SampleIndex) -> Path:
    base = cfg["paths"]["train_audio_dir"] if sample.split == "train" else cfg["paths"]["val_audio_dir"]
    suffix = str(cfg["audio"]["file_suffix"])
    return Path(base) / f"Subject_{sample.subject}_Story_{sample.story}{suffix}"


def feature_path(cfg: dict[str, Any], sample: SampleIndex) -> Path:
    return Path(cfg["paths"]["feature_dir"]) /  f"Subject_{sample.subject}_Story_{sample.story}_aligned.npy"


def read_labels(cfg: dict[str, Any], sample: SampleIndex) -> np.ndarray:
    ann = annotation_path(cfg, sample)
    if not ann.exists():
        raise FileNotFoundError(f"Annotation missing: {ann}")
    y = pd.read_csv(ann).iloc[:, 0].to_numpy(dtype=np.float32)
    if y.ndim != 1:
        y = y.reshape(-1)
    return y


def read_features(cfg: dict[str, Any], sample: SampleIndex) -> np.ndarray:
    feat = feature_path(cfg, sample)
    if not feat.exists():
        raise FileNotFoundError(f"Feature missing: {feat}")
    x = np.load(feat)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D features in {feat}, got shape {x.shape}")
    return x.astype(np.float32)


def load_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    wav_path = Path(path)
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio missing: {wav_path}")

    with wave.open(str(wav_path), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sampwidth == 1:
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        arr = (arr - 128.0) / 128.0
    elif sampwidth == 2:
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth}")

    if channels > 1:
        arr = arr.reshape(-1, channels).mean(axis=1)
    return arr.astype(np.float32), int(sample_rate)


def extract_torch_features(waveform: np.ndarray, sample_rate: int, cfg: dict[str, Any]) -> np.ndarray:
    if torch is None or F is None:
        raise ImportError("PyTorch is required for feature extraction.")
    feature_cfg = cfg["feature"]
    target_sr = int(cfg["audio"]["sample_rate"])
    n_fft = int(feature_cfg["n_fft"])
    hop = int(feature_cfg["hop_length"])
    win = int(feature_cfg["win_length"])
    n_freq_bins = int(feature_cfg["n_freq_bins"])
    log_offset = float(feature_cfg["log_offset"])

    wav = torch.tensor(waveform, dtype=torch.float32)
    if sample_rate != target_sr:
        # Linear resampling keeps dependencies minimal and stays in torch.
        wav = wav.view(1, 1, -1)
        new_len = int(round((wav.shape[-1] * target_sr) / float(sample_rate)))
        wav = F.interpolate(wav, size=new_len, mode="linear", align_corners=False).view(-1)

    window = torch.hann_window(win)
    spec = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=window,
        center=True,
        return_complex=True,
    )
    mag = torch.abs(spec)
    if n_freq_bins < mag.shape[0]:
        mag = mag[:n_freq_bins]
    feats = torch.log(mag + log_offset).transpose(0, 1).contiguous()
    return feats.cpu().numpy().astype(np.float32)


def align_features_to_frames(features: np.ndarray, num_frames: int) -> np.ndarray:
    if num_frames <= 0:
        return np.zeros((0, features.shape[1]), dtype=np.float32)
    if len(features) == num_frames:
        return features.astype(np.float32)
    if torch is None or F is None:
        raise ImportError("PyTorch is required for feature alignment.")

    x = torch.tensor(features, dtype=torch.float32).transpose(0, 1).unsqueeze(0)
    aligned = F.interpolate(x, size=num_frames, mode="linear", align_corners=False)
    return aligned.squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32)


def build_aligned_features(cfg: dict[str, Any], sample: SampleIndex) -> Path:
    audio = audio_path(cfg, sample)
    labels = read_labels(cfg, sample)
    wav, sr = load_wav_mono(audio)
    feats = extract_torch_features(wav, sr, cfg)
    aligned = align_features_to_frames(feats, len(labels))
    out = feature_path(cfg, sample)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, aligned)
    return out


def window_sequence(
    x: np.ndarray,
    y: np.ndarray,
    window_size: int,
    stride: int,
    include_last: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    if n < window_size:
        return (
            np.empty((0, window_size, x.shape[-1]), dtype=np.float32),
            np.empty((0, window_size), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    starts = list(range(0, n - window_size + 1, stride))
    last_start = n - window_size
    if include_last and (not starts or starts[-1] != last_start):
        starts.append(last_start)

    x_windows = np.asarray([x[s : s + window_size] for s in starts], dtype=np.float32)
    y_windows = np.asarray([y[s : s + window_size] for s in starts], dtype=np.float32)
    return x_windows, y_windows, np.asarray(starts, dtype=np.int32)


def reconstruct_from_windows(pred_windows: np.ndarray, starts: np.ndarray, total_len: int) -> np.ndarray:
    acc = np.zeros((total_len,), dtype=np.float32)
    cnt = np.zeros((total_len,), dtype=np.float32)

    for win, start in zip(pred_windows, starts):
        end = min(total_len, start + len(win))
        w = win[: end - start]
        acc[start:end] += w
        cnt[start:end] += 1.0

    cnt[cnt == 0.0] = 1.0
    return acc / cnt


class SpeechWindowDataset(Dataset):
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
            raise ImportError("PyTorch is required for SpeechWindowDataset.")

        m = cfg["model"]
        window_size = int(m["sequence_length"])
        stride = int(m["stride"])

        x_rows: list[np.ndarray] = []
        y_rows: list[np.ndarray] = []
        s_rows: list[int] = []

        for sample in iter_samples(cfg, split):
            try:
                x = read_features(cfg, sample)
                y = read_labels(cfg, sample)
            except FileNotFoundError:
                continue
            xw, yw, _ = window_sequence(x, y, window_size=window_size, stride=stride, include_last=True)
            for xv, yv in zip(xw, yw):
                x_rows.append(xv)
                y_rows.append(yv)
                s_rows.append(sample.subject - 1)

        if x_rows:
            x_np = np.asarray(x_rows, dtype=np.float32)
            y_np = np.asarray(y_rows, dtype=np.float32)
            s_np = np.asarray(s_rows, dtype=np.int64)
        else:
            feat_dim = int(cfg["feature"]["n_freq_bins"])
            x_np = np.empty((0, window_size, feat_dim), dtype=np.float32)
            y_np = np.empty((0, window_size), dtype=np.float32)
            s_np = np.empty((0,), dtype=np.int64)

        self.feature_mean = float(np.mean(x_np)) if feature_mean is None else float(feature_mean)
        self.feature_std = float(np.std(x_np)) if feature_std is None else float(feature_std)
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
        self.subject_idx = s_np

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.x[idx], dtype=torch.float32),
            torch.tensor(self.subject_idx[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


class SpeechBiGRUModel(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        if torch is None or nn is None:
            raise ImportError("PyTorch is required for SpeechBiGRUModel.")
        super().__init__()
        m = cfg["model"]
        input_dim = int(cfg["feature"]["n_freq_bins"])
        hidden_dim = int(m["hidden_dim"])
        num_layers = int(m["num_layers"])
        dropout = float(m["dropout"])
        self.subject_embed_dim = int(m["subject_embed_dim"])

        all_subjects = set(cfg["split"]["subjects_train"]) | set(cfg["split"]["subjects_val"])
        subject_count = max(all_subjects)

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

        if self.subject_embed_dim > 0:
            self.subject_embed = nn.Embedding(subject_count, self.subject_embed_dim)
        else:
            self.subject_embed = None

        head_in = hidden_dim * 2 + max(0, self.subject_embed_dim)
        self.head = nn.Linear(head_in, 1)

    def forward(self, x: torch.Tensor, subject_idx: torch.Tensor) -> torch.Tensor:
        seq, _ = self.gru(x)
        seq = self.dropout(seq)
        if self.subject_embed is not None:
            sid = self.subject_embed(subject_idx).unsqueeze(1).expand(-1, seq.shape[1], -1)
            seq = torch.cat([seq, sid], dim=-1)
        out = self.head(seq).squeeze(-1)
        return out


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
    return Path(cfg["paths"]["checkpoint_dir"]) / "speech_bigru.pt"


def now_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
