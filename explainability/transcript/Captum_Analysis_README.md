# Transcript Model — Captum Explainability

Integrated Gradients attribution analysis for the Transcript LSTM model.  
Produces three plots showing which of the 11 lexicon features drive valence predictions.

---

## Prerequisites

### 1. Python Environment

The `.venv` is located inside the `transcript/` folder. Activate it before running anything.

**Fix PowerShell execution policy (one-time, if not already done):**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Activate the venv:**
```powershell
cd "D:\Final_Project\S4 Project\omg-refactor\src\transcript"
& ".venv\Scripts\Activate.ps1"
```

You should see the venv prefix appear in your prompt, e.g. `(omg-challenge20...)`.

### 2. Install Dependencies

With the venv active, install all required packages:

```powershell
python -m pip install captum pyyaml pandas numpy torch pyarrow
```

Verify captum installed correctly:
```powershell
python -c "from captum.attr import IntegratedGradients; print('captum OK')"
```

---

## Config Setup

The standard `config.yaml` uses relative paths and validates input directories that may not
exist (e.g. `srt_dir`). Use the dedicated `explain_config.yaml` instead, which uses absolute
paths and skips input directory validation.

Create `explain_config.yaml` in the `transcript/` folder with the following content,
adjusting paths to match your machine:

```yaml
paths:
  srt_dir: D:\Final_Project\S3 Project\OmgEmapathyPlus-master\Annotations\Transcriptions
  train_ann_dir: D:\Final_Project\S3 Project\Training\Annotations
  val_ann_dir: D:\Final_Project\S3 Project\Validation\Annotations
  lexicon_dir: D:\Final_Project\S4 Project\omg-refactor\src\raw_data\transcript\lexicons
  feature_dir: D:\Final_Project\S4 Project\omg-refactor\src\transcript\artifacts\features
  checkpoint_dir: D:\Final_Project\S4 Project\omg-refactor\src\transcript\artifacts\checkpoints
  prediction_dir: D:\Final_Project\S4 Project\omg-refactor\src\transcript\artifacts\predictions

split:
  manifest_id: transcript_v1
  subjects_train: [1,2,3,4,5,6,7,8,9,10]
  subjects_val: [1,2,3,4,5,6,7,8,9,10]
  stories_train: [1,4,5,8]
  stories_val: [2]

model:
  window_size: 100
  stride: 50
  embedding_size: 11
  lstm_hidden_dim: 64
  subject_embed_dim: 2
  dense_hidden_dim: 32
  dropout: 0.2

train:
  epochs: 20
  batch_size: 256
  lr: 0.0001
  patience: 5
  device: auto

predict:
  batch_size: 512
  device: auto
```

---

## Step 1 — Generate Feature Files (if not already done)

The explainability script requires preprocessed `.npy` feature files in `artifacts/features/`.
If this folder is empty, run preprocessing first:

```powershell
python preprocess.py --config explain_config.yaml
```

This takes approximately 5–10 minutes. When complete you should see output like:
```
Done. Built 35 aligned features, skipped 5.
```

Verify the files were created:
```powershell
dir "D:\Final_Project\S4 Project\omg-refactor\src\transcript\artifacts\features"
```

You should see files named `Subject_X_Story_Y_aligned.npy` for all subjects and stories.

---

## Step 2 — Run Explainability

### Standard run (all training stories, all subjects)

```powershell
python transcript_explain.py --config explain_config.yaml --story 0
```

`--story 0` disables the story filter and uses all available feature files.

### Held-out validation analysis (Story 2 only)

To run attribution on the held-out validation set that the model never trained on:

```powershell
python transcript_explain.py --config explain_config.yaml --story 2
```

> Note: This requires Story 2 `.npy` files to exist in `artifacts/features/`.
> If they are missing, run `preprocess.py` with `val_ann_dir` pointing to the
> Story 2 annotation files and ensure Story 2 SRT files are available.

### Restrict to a specific subject

```powershell
python transcript_explain.py --config explain_config.yaml --story 0 --subject 3
```

### All available flags

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `transcript/config.yaml` | Path to config file |
| `--checkpoint` | from config | Override checkpoint path manually |
| `--output-dir` | `transcript/artifacts/explanations` | Where to save plots and .npy files |
| `--top-n` | `11` | Number of features shown in bar chart |
| `--max-samples` | `200` | Max windows used for IG (more = slower but more stable) |
| `--subject` | `0` | Filter to one subject (0 = all subjects) |
| `--story` | `2` | Filter to one story (0 = all stories) |
| `--n-steps` | `50` | Integrated Gradients integration steps |
| `--batch-size` | `64` | Batch size for IG computation |
| `--device` | `auto` | Device: auto / cpu / cuda / mps |

---

## Output Files

All outputs are saved to `transcript/artifacts/explanations/` by default.

| File | Description |
|------|-------------|
| `captum_transcript_attributions.png` | Top-N horizontal bar chart — mean absolute attribution per feature |
| `captum_transcript_signed.png` | Signed mean attribution — green = positive, red = negative contribution |
| `captum_transcript_temporal_heatmap.png` | Heatmap of attribution per feature across the 100-timestep window |
| `ig_attributions.npy` | Raw attribution array of shape `(N, T, F)` — samples × timesteps × features |
| `mean_abs_per_feature.npy` | Mean absolute attribution per feature, shape `(11,)` |

---

## Feature Names

The 11 input features in order (indices 0–10):

| Index | Feature | Source |
|-------|---------|--------|
| 0 | Warriner: Valence | Warriner et al. lexicon |
| 1 | Warriner: Arousal | Warriner et al. lexicon |
| 2 | Warriner: Dominance | Warriner et al. lexicon |
| 3 | DepecheMood: Afraid | DepecheMood lexicon |
| 4 | DepecheMood: Amused | DepecheMood lexicon |
| 5 | DepecheMood: Angry | DepecheMood lexicon |
| 6 | DepecheMood: Annoyed | DepecheMood lexicon |
| 7 | DepecheMood: Dont_Care | DepecheMood lexicon |
| 8 | DepecheMood: Happy | DepecheMood lexicon |
| 9 | DepecheMood: Inspired | DepecheMood lexicon |
| 10 | DepecheMood: Sad | DepecheMood lexicon |

---

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `essentia wheel incompatibility` | Python 3.14 used | `uv python pin 3.13` then `uv add captum` |
| `running scripts is disabled` | PowerShell execution policy | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `No module named yaml` | pyyaml not installed | `python -m pip install pyyaml` |
| `Input directory missing: srt_dir` | Relative paths in config | Use `explain_config.yaml` with absolute paths |
| `No validation windows found` | story filter defaulting to 2 | Run with `--story 0` |
| `cudnn RNN backward can only be called in training mode` | Model in eval() during IG | Already fixed in script — model.train() called inside IG function |
| `from captum.attr import IntegratedGradients` fails | captum installed in wrong env | `python -m pip install captum` (not just `pip install`) |

---

## Notes

- Attributions are computed on **training stories (1, 4, 5, 8)** by default since Story 2
  feature files may not be available. This does not affect the validity of feature importance
  results — the model weights are fixed and IG measures input sensitivity regardless of split.
- For a **generalisation check**, run separately on Story 2 (held-out validation set) and
  compare feature rankings. Consistent rankings across training and validation stories indicate
  the attribution patterns are not artefacts of overfitting.
- The `--max-samples 200` default balances speed and stability. Increasing to 500+ will give
  smoother attribution estimates but takes proportionally longer.
- CUDA is used automatically if available (`--device auto`). CPU is significantly slower
  for IG due to the number of forward passes required (n_steps × batch_size).
