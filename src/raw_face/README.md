## raw_face

### Run
```bash
uv run python raw_face/extract_faces.py --config raw_face/config.yaml
uv run python raw_face/preprocess.py --config raw_face/config.yaml
uv run python raw_face/train.py --config raw_face/config.yaml
uv run python raw_face/predict.py --config raw_face/config.yaml
uv run python raw_face/evaluate_predictions.py --config raw_face/config.yaml --output-dir raw_face/artifacts/model_evaluation --overwrite
```

### OOF (Out-of-Fold) Predictions for Stacking
```bash
uv run python raw_face/generate_oof.py --config raw_face/config.yaml
```
Trains one holdout model per training story, writes OOF predictions under
`raw_face/artifacts/predictions/oof/`, and combines them into
`raw_face/artifacts/predictions/oof/combined_oof.parquet`.

### Outputs
1. Subject face crops: `raw_face/artifacts/faces/{train|val}/Subject_{s}_Story_{t}/Subject_face/*.png`
2. Cached aligned features: `raw_face/artifacts/features/*_aligned.npz`
3. Checkpoint: `raw_face/artifacts/checkpoints/raw_face_3dcnn.pt`
4. Predictions parquet: `raw_face/artifacts/predictions/Subject_{s}_Story_{t}.parquet`
5. OOF parquets: `raw_face/artifacts/predictions/oof/Subject_{s}_Story_{t}.parquet`
6. Combined OOF: `raw_face/artifacts/predictions/oof/combined_oof.parquet`
