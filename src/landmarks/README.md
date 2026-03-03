## landmarks

A clean-slate PyTorch landmarks pipeline isolated from legacy TensorFlow scripts.

### Run
```bash
uv run python landmarks/extract_landmarks.py --config landmarks/config.yaml
uv run python landmarks/preprocess.py --config landmarks/config.yaml
uv run python landmarks/train.py --config landmarks/config.yaml
uv run python landmarks/predict.py --config landmarks/config.yaml
uv run python landmarks/evaluate_predictions.py --config landmarks/config.yaml --output-dir landmarks/artifacts/model_evaluation
```

### Input
Landmark CSV files from:
1. `paths.train_landmarks_csv_dir`
2. `paths.val_landmarks_csv_dir`

The extraction command writes:
`{train|val}_landmarks_csv_dir/Subject_{subject}_Story_{story}/Subject_face_landmarks.csv`

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
