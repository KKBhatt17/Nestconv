"""Stage 1: rearrange a (BLIP-derived) ViT checkpoint by unit importance.

Reorders attention heads and MLP neurons so the most important units come first,
which is what makes nested (prefix) subnetworks valid for the elastic backbone.
Pure tensor surgery on the state dict; no training.

The input checkpoint is a timm-style ViT state dict. For a BLIP vision tower,
produce it first with
``elastic_vqa.models.blip_vision_adapter.export_blip_vision_checkpoint``.
"""

from __future__ import annotations

from pprint import pformat
from typing import Dict

from elastic_vqa.models.blip_vision_adapter import export_blip_vision_checkpoint
from elastic_vqa.models.checkpoint_ops import rearrange_vit_checkpoint


def run_rearrangement(config: Dict) -> None:
    step_cfg = config["stage1"]
    input_checkpoint = step_cfg.get("input_checkpoint")

    # Optionally derive the timm-style input checkpoint straight from a BLIP model.
    blip_export = step_cfg.get("export_from_blip")
    if blip_export:
        input_checkpoint = export_blip_vision_checkpoint(
            blip_model_name=blip_export["blip_model_name"],
            output_path=blip_export["output_path"],
            num_layers=step_cfg.get("num_layers", 12),
            target_pos_tokens=blip_export.get("target_pos_tokens"),
        )
        print(f"Exported BLIP vision tower to timm checkpoint: {input_checkpoint}")

    _, meta = rearrange_vit_checkpoint(
        input_path=input_checkpoint,
        output_path=step_cfg["output_checkpoint"],
        num_layers=step_cfg.get("num_layers", 12),
        embed_dim=step_cfg.get("embed_dim", 768),
        num_heads=step_cfg.get("max_heads", 12),
    )
    print("Rearranged checkpoint saved.")
    print(pformat({k: f"<{len(v)} layers>" for k, v in meta.items()}))
