# Elastic Nested ViT for VQA

A two-stage pipeline that puts a **width-elastic ViT vision tower** under a BLIP
VQA model and curriculum-trains it, then evaluates at a fixed set of backbone
presets. Adapted from the Elastic-ViT classification codebase, keeping exactly two
of its mechanisms — the **elastic backbone** and its **curriculum (sandwich)
training** — and dropping the entropy router and entropy-sorted loading (those
existed only to serve the router).

## Pipeline

```bash
pip install -r requirements.txt

# Stage 1: export BLIP's vision tower to a timm-style checkpoint and rearrange it
#          (reorder heads/MLP neurons by importance so nested subnetworks are valid)
python -m elastic_vqa.cli rearrange     --config configs/gqa.yaml

# Stage 2: curriculum elastic training + fixed-preset evaluation each epoch
python -m elastic_vqa.cli train-elastic --config configs/gqa.yaml
```

Smoke-test the whole flow with no real data first:

```bash
python -m elastic_vqa.cli train-elastic --config configs/dummy.yaml
```

## Model

- **Vision:** elastic ViT-B/16 (`elastic_vqa/models/elastic_vit.py`), initialized
  from BLIP-base's vision encoder. Subnetworks are realized by masking, so one
  module evaluates at any preset.
- **Text + fusion:** BLIP's multimodal text encoder. The question tokens
  cross-attend to the elastic vision tokens. Frozen by default
  (`stage2.freeze_text: true`); set to `false` to fine-tune.
- **Answering:** classification over a fixed answer vocab (`data/vocab.py`), top-K
  most frequent training answers. This is what lets the curriculum loop and
  preset eval carry over from the classification codebase.

## Supported datasets

| Dataset | `answer_vocab_size` | Notes |
|---|---|---|
| GQA | ~1842 | `train_balanced` / `testdev_balanced`; single ground-truth answer |
| CLEVR | ~28 | Closed answer set → no vocab-coverage cap; good fast smoke test |
| dummy | 16 | Synthetic; verifies the pipeline with no data/GPU |

## Metrics (per eval preset)

- **Exact-match accuracy** — primary; out-of-vocab answers count as incorrect, so
  the headline number is over the *full* eval set (vocab-coverage cap included).
- **Validation CE loss.**
- **Vision-tower MACs** — makes the accuracy-vs-compute trade-off across presets
  legible. (`estimate_vit_macs` uses a default token count; for 384px set
  `tokens=577` if you want absolute MACs rather than relative.)

Open/Binary accuracy split and GQA's structured diagnostics (consistency,
validity, plausibility) are deliberately out of scope for v1 — they need the
official GQA scorer and question metadata.

## BLIP integration — verify first

`models/blip_vision_adapter.py` remaps BLIP's HuggingFace vision encoder to timm
ViT key names so `rearrange` and the elastic backbone can consume it. This is the
one piece of real glue: run `audit_key_mapping` once against your installed
`transformers` version to confirm no keys are dropped. BLIP-vqa-base runs at
384px (577 tokens) → pair with `vit_base_patch16_384` and `image_size: 384`.
Rearrange is valid because BLIP fusion consumes the final post-norm vision tokens
(not intermediate per-neuron features).
