## transcript

### Run
```bash
uv run python transcript/preprocess.py --config transcript/config.yaml
uv run python transcript/train.py --config transcript/config.yaml
uv run python transcript/predict.py --config transcript/config.yaml
```

### Tests
```bash
uv run -m unittest discover -s transcript/tests -p "test_*.py"
```

### Config
`transcript/config.yaml` contains:
1. `paths`: all input/output directories.
2. `split`: train/val subject+story lists and `manifest_id`.
3. `model`: architecture/windowing values.
4. `train`: epochs, batch size, lr, patience, device.
5. `predict`: batch size, device.

### Output
Parquet only:
`<prediction_dir>/Subject_{subject}_Story_{story}.parquet`

Required columns:
1. `window_idx`
2. `window_start_frame`
3. `window_end_frame`
4. `window_center_frame`
5. `window_center_s`
6. `y_pred`
7. `subject_id`
8. `story_id`
9. `split`
10. `manifest_id`
