from __future__ import annotations

from pprint import pformat

from elastic_vit_retrieval.models.checkpoint_ops import rearrange_clip_checkpoint


def run_rearrangement(config) -> None:
    pretrained_cfg = config["pretrained"]
    step_cfg = config["stage1"]
    _, meta = rearrange_clip_checkpoint(
        model_name=pretrained_cfg["model_name"],
        output_path=step_cfg["output_checkpoint"],
        cache_dir=pretrained_cfg.get("cache_dir"),
        num_layers=step_cfg.get("num_layers", 12),
        embed_dim=step_cfg.get("embed_dim", 768),
        num_heads=step_cfg.get("max_heads", 12),
    )
    print("Rearranged CLIP checkpoint saved.")
    print(pformat(meta))
