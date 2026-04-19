## ensemble_fusion

### Run
```bash
uv run python ensemble_fusion/train.py --config ensemble_fusion/config.yaml
uv run python ensemble_fusion/predict.py --config ensemble_fusion/config.yaml
uv run python ensemble_fusion/evaluate_predictions.py --config ensemble_fusion/config.yaml --output-dir ensemble_fusion/artifacts/model_evaluation --overwrite
```

### Model

This package trains a simple stacking meta-learner over the five modality predictions:

- speech
- transcript
- landmarks
- raw_face
- fullbody

The current implementation uses raw aligned modality predictions only and fits `scikit-learn` `ElasticNet(positive=True)`.

### Inputs

- training uses each modality's `oof/Subject_{s}_Story_{t}.parquet`
- prediction uses each modality's standard val prediction parquet

### Outputs

1. Checkpoint: `ensemble_fusion/artifacts/checkpoints/elastic_net_positive.pkl`
2. Predictions parquet: `ensemble_fusion/artifacts/predictions/Subject_{s}_Story_{t}.parquet`
3. Evaluation reports: `ensemble_fusion/artifacts/model_evaluation/*`
