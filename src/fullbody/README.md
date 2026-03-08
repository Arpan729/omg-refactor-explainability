## fullbody

### Run
```bash
uv run python fullbody/extract_fullbody.py --config fullbody/config.yaml
uv run python fullbody/preprocess.py --config fullbody/config.yaml
uv run python fullbody/train.py --config fullbody/config.yaml
uv run python fullbody/predict.py --config fullbody/config.yaml
uv run python fullbody/evaluate_predictions.py --config fullbody/config.yaml --output-dir fullbody/artifacts/model_evaluation --overwrite
```

### Outputs
1. Extracted crops: `fullbody/artifacts/fullbody/{train|val}/Subject_{s}_Story_{t}/{Actor_img|Subject_img}/*.png`
2. Cached aligned features: `fullbody/artifacts/features/Subject_{s}_Story_{t}_aligned.npz`
3. Checkpoint: `fullbody/artifacts/checkpoints/fullbody_resnet3d.pt`
4. Predictions parquet: `fullbody/artifacts/predictions/Subject_{s}_Story_{t}.parquet`
