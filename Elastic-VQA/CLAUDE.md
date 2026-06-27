# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Elastic Nested ViT for VQA: a two-stage pipeline that puts a width-elastic ViT
vision tower under a BLIP VQA model, curriculum-trains it, and evaluates it at a
fixed set of backbone presets. Derived from the Elastic-ViT classification
codebase but keeping only the **elastic backbone** and its **curriculum training**;
the router stage and entropy-sorted loading are intentionally absent.

## Commands

Run from the `Elastic-VQA/` directory.

```bash
pip install -r requirements.txt
python -m elastic_vqa.cli rearrange     --config configs/gqa.yaml   # stage 1
python -m elastic_vqa.cli train-elastic --config configs/gqa.yaml   # stage 2
python -m elastic_vqa.cli train-elastic --config configs/dummy.yaml # no-data smoke test
```

There is no test suite, linter config, or build step.

## Configuration

- `configs/gqa.yaml` is the base; `clevr.yaml` and `dummy.yaml` inherit via
  `base_config:` and override only what differs (recursive deep-merge in
  `config.py`). `base_config` resolves relative to the config file's directory.
- Stages communicate through file paths: `stage1.output_checkpoint` →
  `stage2.vision_checkpoint`. Stage 2 writes `vqa_elastic_epoch_{N}.pt`.
- `dataset.root` paths are placeholders and must be set before running.

## Architecture

### Two stages
1. **rearrange** (`engine/rearrange.py`, `models/checkpoint_ops.py`) — optionally
   exports BLIP's vision tower to a timm-style checkpoint
   (`models/blip_vision_adapter.py`), then reorders attention heads and MLP
   neurons by L2 importance so nested (prefix) subnetworks are valid. Pure tensor
   surgery; no training.
2. **train-elastic** (`engine/train_vqa.py`) — curriculum (sandwich) training:
   samples a random per-layer subnetwork each step from the currently unlocked
   width choices (`stage2.curriculum.unlock_fractions`, large-to-small) and trains
   the VQA model under it with cross-entropy. Each epoch ends with fixed-preset
   evaluation (`engine/evaluate_vqa.py`, `stage2.eval_presets`).

### Model (`models/vqa_model.py`)
- `VqaElasticModel` = elastic ViT vision tower + BLIP multimodal text encoder +
  answer head. `forward(pixel_values, input_ids, attention_mask, config=...)`.
- The elastic tower (`models/elastic_vit.py`) returns the **full normalized token
  sequence** via `forward_tokens`; question tokens cross-attend to it inside the
  BLIP text encoder. Pooled `[CLS]` of the fused sequence → linear answer head.
- Subnetworks are realized by **masking**, not slicing, so a single module
  evaluates at any preset. Hard `SubnetworkConfig` is used throughout VQA; the
  soft/router path is retained only for backbone API parity.
- `freeze_text=True` (default) trains only the vision tower + head. The text
  encoder is never run under `no_grad` — gradients must flow through it to the
  vision tower; freezing is via `requires_grad=False`.

### Data (`data/`)
- VQA-as-classification over a fixed answer vocab (`data/vocab.py`); answers
  outside the vocab map to `UNANSWERABLE = -1`, ignored by the CE loss and counted
  wrong by accuracy. All datasets share `task_type == "vqa"`.
- `gqa.py`, `clevr.py`, `dummy.py` each yield `(image, question, answer_index)`;
  `datasets.py` holds the registry, BLIP tokenizer, and the collate fn that stacks
  images, tokenizes questions, and stacks labels.
- `okvqa.py` (OK-VQA) has VQAv2-style annotations (separate question/annotation
  JSONs merged by `question_id`, 10 human answers/question over COCO images). It
  yields a 4-tuple `(image, question, answer_index, answer_idxs)` — the extra
  `answer_idxs` is the 10 human answers as vocab indices. The CE training label is
  the most-frequent (officially-normalized) answer; eval branches on the optional
  `answer_targets` collate key to score the official VQA soft accuracy
  (`utils/metrics.py:vqa_soft_accuracy`). The 4th item field and `answer_targets`
  key are additive — single-answer datasets are unaffected. Optional
  `dataset.images_root` points the (standard COCO) images outside `root`, overriding
  the default `<root>/images` base.

## Conventions
- All modules use `from __future__ import annotations` and type hints.
- Configs are plain dicts; access via `config["stage1"|"stage2"]`,
  `config["dataset"]`, `config["runtime"]`.
- Checkpoints saved per-epoch as `vqa_elastic_epoch_{N}.pt` with embedded
  training `history`; the answer vocab is also written to the output dir.

## Integration caveat
The BLIP→timm vision key remap (`models/blip_vision_adapter.py`) is the one piece
of glue to verify against your `transformers` version — use `audit_key_mapping`.
BLIP-vqa-base is 384px (577 tokens): pair with `vit_base_patch16_384`.
