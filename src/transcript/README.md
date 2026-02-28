## transcript_next

A clean-slate transcript pipeline isolated from legacy repo configs.

### Principles
1. No dependency on `configs/defaults.yaml` or old split manifests.
2. Single source of truth: `transcript_next/config.yaml`.
3. Three commands only.

### Run
```bash
uv run python transcript_next/preprocess.py --config transcript_next/config.yaml
uv run python transcript_next/train.py --config transcript_next/config.yaml
uv run python transcript_next/predict.py --config transcript_next/config.yaml
```

### Config
`transcript_next/config.yaml` contains:
1. `paths`: all input/output directories.
2. `split`: train/val subject+story lists and `manifest_id`.
3. `model`: architecture/windowing values.
4. `train`: epochs, batch size, lr, patience, device.
5. `predict`: batch size, device.

### Output
Parquet only:
`<prediction_dir>/Subject_{subject}_Story_{story}.parquet`

Required columns:
1. `frame_idx`
2. `timestamp_s`
3. `y_pred`
4. `subject_id`
5. `story_id`
6. `split`
7. `manifest_id`
