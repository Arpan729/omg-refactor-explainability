## speech

### Run
```bash
uv run python speech/preprocess.py --config speech/config.yaml
uv run python speech/train.py --config speech/config.yaml
uv run python speech/predict.py --config speech/config.yaml
```

### Config
`speech/config.yaml` contains:
1. `paths`: all input/output directories.
2. `split`: train/val subject+story lists and `manifest_id`.
3. `audio`: sample rate, fps, and audio suffix.
4. `feature`: STFT parameters.
5. `model`: sequence windowing and BiGRU architecture.
6. `train`: optimization and runtime settings.
7. `predict`: runtime batch settings.

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
