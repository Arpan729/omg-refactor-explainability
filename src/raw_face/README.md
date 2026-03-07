## raw_face

A clean-slate PyTorch raw-face pipeline with modality-owned face artifacts.

### Run
```bash
uv run python raw_face/extract_faces.py --config raw_face/config.yaml
uv run python raw_face/preprocess.py --config raw_face/config.yaml
uv run python raw_face/train.py --config raw_face/config.yaml
uv run python raw_face/predict.py --config raw_face/config.yaml
uv run python raw_face/evaluate_predictions.py --config raw_face/config.yaml --output-dir raw_face/artifacts/model_evaluation --overwrite
```

### Outputs
1. Subject face crops: `raw_face/artifacts/faces/{train|val}/Subject_{s}_Story_{t}/Subject_face/*.png`
2. Cached aligned features: `raw_face/artifacts/features/*_aligned.npz`
3. Checkpoint: `raw_face/artifacts/checkpoints/raw_face_3dcnn.pt`
4. Predictions parquet: `raw_face/artifacts/predictions/Subject_{s}_Story_{t}.parquet`
