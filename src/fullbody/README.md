## fullbody

### Run
```bash
uv run python fullbody/extract_fullbody.py --config fullbody/config.yaml
uv run python fullbody/preprocess.py --config fullbody/config.yaml
uv run python fullbody/train.py --config fullbody/config.yaml
uv run python fullbody/predict.py --config fullbody/config.yaml
uv run python fullbody/evaluate_predictions.py --config fullbody/config.yaml --output-dir fullbody/artifacts/model_evaluation --overwrite
```

### OOF (Out-of-Fold) Predictions for Stacking
```bash
uv run python fullbody/generate_oof.py --config fullbody/config.yaml
```
Trains one holdout model per training story, writes OOF predictions under
`fullbody/artifacts/predictions/oof/`, and combines them into
`fullbody/artifacts/predictions/oof/combined_oof.parquet`.

### Outputs
1. Extracted crops: `fullbody/artifacts/fullbody/{train|val}/Subject_{s}_Story_{t}/{Actor_img|Subject_img}/*.png`
2. Cached aligned features: `fullbody/artifacts/features/Subject_{s}_Story_{t}_aligned.npz`
3. Checkpoint: `fullbody/artifacts/checkpoints/fullbody_resnet3d.pt`
4. Predictions parquet: `fullbody/artifacts/predictions/Subject_{s}_Story_{t}.parquet`
5. OOF parquets: `fullbody/artifacts/predictions/oof/Subject_{s}_Story_{t}.parquet`
6. Combined OOF: `fullbody/artifacts/predictions/oof/combined_oof.parquet`
