# Ensemble fusion explainability

Captum explainability for the saved ensemble-fusion stacker.

This script explains the raw stacked model output from the checkpoint. It does
not explain the final smoothed post-processing path.

The stacker is linear, so one Integrated Gradients step is enough: the gradient
is constant along the path from the zero baseline to each input.

Run from the `src` uv environment:

```bash
cd src
uv run python ../explainability/ensemble_fusion/explain_ensemble_fusion.py \
  --config ../explainability/ensemble_fusion/config.yaml
```

Outputs are written to `explainability/ensemble_fusion/artifacts`:

- `captum_ensemble_global_importance.csv`
- `captum_ensemble_subject_importance.csv`
- `captum_ensemble_attributions.npz`
- `captum_ensemble_global_importance.png`
- `captum_ensemble_subject_importance.png`
