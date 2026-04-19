from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from sklearn.linear_model import ElasticNet
import torch
import torch.nn.functional as F
import yaml


REQUIRED_TOP_LEVEL_KEYS = ["paths", "split", "fusion", "train", "predict", "model"]
REQUIRED_PATH_KEYS = ["train_ann_dir", "val_ann_dir", "checkpoint_dir", "prediction_dir_raw", "prediction_dir_smoothed"]
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
    oof_dir: Path
    kind: str
    cutoff: float
    order: int


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
        for field in ["prediction_dir", "oof_dir"]:
            if field not in modality_cfg:
                raise ValueError(f"Missing fusion.modalities.{name}.{field}")


def _resolve_paths(cfg: dict[str, Any]) -> None:
    for key in REQUIRED_PATH_KEYS:
        p = Path(cfg["paths"][key])
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        cfg["paths"][key] = str(p)

    for modality_cfg in cfg["fusion"]["modalities"].values():
        for field in ["prediction_dir", "oof_dir"]:
            p = Path(modality_cfg[field])
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            modality_cfg[field] = str(p)


def _validate_dirs(cfg: dict[str, Any]) -> None:
    for key in ["train_ann_dir", "val_ann_dir"]:
        p = Path(cfg["paths"][key])
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Input directory missing: paths.{key}={p}")
    for name, modality_cfg in cfg["fusion"]["modalities"].items():
        for field in ["prediction_dir", "oof_dir"]:
            p = Path(modality_cfg[field])
            if not p.exists() or not p.is_dir():
                raise FileNotFoundError(f"Input directory missing: fusion.modalities.{name}.{field}={p}")


def _ensure_output_dirs(cfg: dict[str, Any]) -> None:
    Path(cfg["paths"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["prediction_dir_raw"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["prediction_dir_smoothed"]).mkdir(parents=True, exist_ok=True)


def _validate_split(cfg: dict[str, Any]) -> None:
    train_stories = set(cfg["split"]["stories_train"])
    val_stories = set(cfg["split"]["stories_val"])
    overlap = train_stories.intersection(val_stories)
    if overlap:
        raise ValueError(f"Train/val story overlap detected: {sorted(overlap)}")


def choose_device(device_flag: str):
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


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    return pd.read_csv(path).iloc[:, 0].to_numpy(dtype=np.float32).reshape(-1)


def modality_configs(cfg: dict[str, Any]) -> list[ModalityConfig]:
    out: list[ModalityConfig] = []
    for name, item in cfg["fusion"]["modalities"].items():
        out.append(
            ModalityConfig(
                name=name,
                prediction_dir=Path(str(item["prediction_dir"])),
                oof_dir=Path(str(item["oof_dir"])),
                kind=str(item.get("kind", "frame")),
                cutoff=float(item.get("cutoff", 0.0)),
                order=int(item.get("order", 1)),
            )
        )
    return out


def checkpoint_path(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["checkpoint_dir"]) / str(cfg["train"].get("checkpoint_name", "elastic_net_positive.pkl"))


def _prediction_file(base_dir: Path, sample: SampleIndex) -> Path:
    return base_dir / f"Subject_{sample.subject}_Story_{sample.story}.parquet"


def prediction_dir_for_variant(cfg: dict[str, Any], variant: str) -> Path:
    if variant == "raw":
        return Path(cfg["paths"]["prediction_dir_raw"])
    if variant == "smoothed":
        return Path(cfg["paths"]["prediction_dir_smoothed"])
    raise ValueError(f"Unsupported prediction variant: {variant}")


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


def read_modality_prediction(modality: ModalityConfig, sample: SampleIndex, target_len: int, source: str) -> np.ndarray:
    if source == "oof":
        base_dir = modality.oof_dir
    elif source == "prediction":
        base_dir = modality.prediction_dir
    else:
        raise ValueError(f"Invalid source: {source}")
    path = _prediction_file(base_dir, sample)
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


def build_feature_frame(cfg: dict[str, Any], sample: SampleIndex, source: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    y_true = read_labels(cfg, sample)
    target_len = len(y_true)
    fps = float(cfg["fusion"].get("fps", 25.0))
    features: list[np.ndarray] = []
    names: list[str] = []
    for modality in modality_configs(cfg):
        preds = read_modality_prediction(modality, sample, target_len=target_len, source=source)
        if len(preds) != target_len:
            raise ValueError(f"Aligned prediction length mismatch for {modality.name}: {len(preds)} vs {target_len}")
        preds = butter_lowpass_filter_bidirectional(preds, cutoff=modality.cutoff, fs=fps, order=modality.order)
        features.append(preds.astype(np.float32))
        names.append(modality.name)
    x = np.stack(features, axis=1) if features else np.zeros((target_len, 0), dtype=np.float32)
    return x, y_true.astype(np.float32), names


def build_training_matrix(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    feature_names: list[str] | None = None
    for sample in iter_samples(cfg, "train"):
        x_sample, y_sample, names = build_feature_frame(cfg, sample, source="oof")
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("Feature names differ across samples")
        xs.append(x_sample)
        ys.append(y_sample)
    if not xs or feature_names is None:
        raise FileNotFoundError("No training OOF samples found for ensemble fusion")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), feature_names


def train_model(cfg: dict[str, Any], checkpoint_out: Path | None = None) -> tuple[Path, dict[str, float]]:
    set_seed(int(cfg["train"]["seed"]))
    x_np, y_np, feature_names = build_training_matrix(cfg)
    model = ElasticNet(
        alpha=float(cfg["train"]["alpha"]),
        l1_ratio=float(cfg["train"]["l1_ratio"]),
        positive=bool(cfg["model"].get("positive", True)),
        fit_intercept=bool(cfg["model"].get("fit_intercept", True)),
        max_iter=int(cfg["model"].get("max_iter", 10000)),
        random_state=int(cfg["train"]["seed"]),
    )
    model.fit(x_np, y_np)

    train_pred = model.predict(x_np).astype(np.float32)
    final_weights = np.asarray(model.coef_, dtype=np.float32)
    final_bias = float(model.intercept_)

    metrics = {
        "train_ccc": float(ccc_numpy(y_np, train_pred)),
        "train_rmse": float(np.sqrt(np.mean((train_pred - y_np) ** 2))),
    }

    ckpt_path = checkpoint_out or checkpoint_path(cfg)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    with ckpt_path.open("wb") as f:
        pickle.dump(
            {
                "model": model,
                "feature_names": list(feature_names),
                "positive": bool(cfg["model"].get("positive", True)),
                "train_metrics": {k: float(v) for k, v in metrics.items()},
                "weights": final_weights.tolist(),
                "bias": final_bias,
                "manifest_id": cfg["split"]["manifest_id"],
            },
            f,
        )
    return ckpt_path, metrics


def predict_sample(
    cfg: dict[str, Any], sample: SampleIndex, ckpt_path: Path
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with ckpt_path.open("rb") as f:
        saved = pickle.load(f)
    model: ElasticNet = saved["model"]

    x_np, _, feature_names = build_feature_frame(cfg, sample, source="prediction")
    if list(feature_names) != list(saved["feature_names"]):
        raise ValueError("Feature order mismatch between config and checkpoint")
    y_pred_raw = model.predict(x_np).astype(np.float32)
    y_pred_smoothed = butter_lowpass_filter_bidirectional(
        y_pred_raw,
        cutoff=float(cfg["fusion"].get("final_cutoff", 0.0)),
        fs=float(cfg["fusion"].get("fps", 25.0)),
        order=int(cfg["fusion"].get("final_order", 1)),
    )
    overlays = {name: x_np[:, idx].astype(np.float32) for idx, name in enumerate(feature_names)}
    return y_pred_raw, y_pred_smoothed, overlays


def write_prediction_parquet(cfg: dict[str, Any], sample: SampleIndex, y_pred: np.ndarray, variant: str) -> Path:
    out_dir = prediction_dir_for_variant(cfg, variant)
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
