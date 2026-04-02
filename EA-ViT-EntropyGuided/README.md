# EA-ViT Entropy-Guided Variant

This directory is a parallel copy of `EA-ViT` where stage 1 and the NSGA bootstrap are preserved, while stage 2 is changed to:

- compute an image entropy mean
- predict a constraint value with an MLP
- pass that predicted constraint to the original router
- select the subnetwork from the router output

The original `EA-ViT` directory is unchanged.

## Stage 2 Changes

- `mean entropy -> constraint predictor MLP -> router -> subnetwork`
- stage 2 begins with a constraint-predictor initialization phase
- the entropy-to-constraint guide loss decays during joint training like the router supervision
- training batches are built from entropy-sorted samples and then batch-shuffled
- validation and inference process one image at a time
- gradient accumulation is supported with `--accum_iter`

## Constraint Guide

Place a CSV such as `constraint_guide.csv` in this directory and pass it with `--constraint_guide_path`.

Required columns:

```csv
EntropyMean,MACs
```

## Training

Stage 1 and the NSGA search scripts are unchanged from the original repo.

For entropy-guided stage 2:

```bash
bash stage2.sh
```

## Evaluation

```bash
bash inference.sh
```

Inference uses single-image routing and predicts the constraint from each image entropy mean before calling the router.
