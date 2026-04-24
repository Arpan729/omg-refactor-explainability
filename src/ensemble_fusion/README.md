## ensemble_fusion

### Run
```bash
uv run python ensemble_fusion/train.py --config ensemble_fusion/config.yaml
uv run python ensemble_fusion/predict.py --config ensemble_fusion/config.yaml
uv run python ensemble_fusion/evaluate_predictions.py --config ensemble_fusion/config.yaml --variant raw --output-dir ensemble_fusion/artifacts/model_evaluation_raw --overwrite
uv run python ensemble_fusion/evaluate_predictions.py --config ensemble_fusion/config.yaml --variant smoothed --output-dir ensemble_fusion/artifacts/model_evaluation_smoothed --overwrite
```

### Model

This package trains a simple stacking meta-learner over the five modality predictions:

- speech
- transcript
- landmarks
- raw_face
- fullbody

The current implementation smooths each modality stream before stacking, fits `scikit-learn` `ElasticNet(positive=True)`, and writes both raw and final-smoothed ensemble outputs for comparison.
The training entrypoint now supports a fast linear-model sweep and promotes the best candidate checkpoint automatically.

### Inputs

- training uses each modality's `oof/Subject_{s}_Story_{t}.parquet`
- prediction uses each modality's standard val prediction parquet

### Outputs

1. Checkpoint: `ensemble_fusion/artifacts/checkpoints/elastic_net_positive.pkl`
2. Raw predictions parquet: `ensemble_fusion/artifacts/predictions_raw/Subject_{s}_Story_{t}.parquet`
3. Smoothed predictions parquet: `ensemble_fusion/artifacts/predictions_smoothed/Subject_{s}_Story_{t}.parquet`
4. Sweep summaries: `ensemble_fusion/artifacts/checkpoints/sweep_summary.{csv,json}`
5. Evaluation reports: `ensemble_fusion/artifacts/model_evaluation_{raw|smoothed}/*`
