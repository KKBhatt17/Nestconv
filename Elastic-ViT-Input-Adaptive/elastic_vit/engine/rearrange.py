from __future__ import annotations

from pprint import pformat

from elastic_vit.models.checkpoint_ops import rearrange_vit_checkpoint


def run_rearrangement(config) -> None:
    step_cfg = config["stage1"]
    _, meta = rearrange_vit_checkpoint(
        input_path=step_cfg["input_checkpoint"],
        output_path=step_cfg["output_checkpoint"],
        num_layers=step_cfg.get("num_layers", 12),
        embed_dim=step_cfg.get("embed_dim", 768),
        num_heads=step_cfg.get("max_heads", 12),
    )
    print("Rearranged checkpoint saved.")
    print(pformat(meta))
