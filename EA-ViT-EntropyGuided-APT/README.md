# EA-ViT Entropy-Guided + Adaptive Patching Variant

This directory is a parallel copy of `EA-ViT-EntropyGuided` with one additional change inspired by `APT`:

- images are no longer always tokenized with a uniform patch grid
- an APT-style entropy-driven adaptive patch selection step now prepares the token sequence before the elastic ViT processes it

The original `APT` and `EA-ViT-EntropyGuided` directories are unchanged.

## What Was Ported From APT

The copied variant adds only the adaptive image-to-token preparation path:

- compute patch entropy maps at multiple patch scales
- select larger patches for low-entropy regions and smaller patches for high-entropy regions
- reuse the original ViT patch embedding weights on the selected patches
- resample position embeddings for larger patch scales
- assemble the final token sequence that is fed into the ViT

The implementation lives in:

- `models/adaptive_patching.py`
- `models/model_stage2.py`

## Current Behavior

This repo still keeps the entropy-guided router from `EA-ViT-EntropyGuided`:

- `mean entropy -> constraint predictor MLP -> router -> subnetwork`

On top of that, each image now uses adaptive patch preparation before entering the ViT.

## Training / Evaluation

Use the same scripts as the entropy-guided variant:

```bash
bash stage2.sh
bash inference.sh
```

The adaptive patching path is enabled inside the copied stage-2 model by default.
