"""Evaluate an Elastic-ViT detector at N fixed backbone sub-networks.

For each sub-network listed in ``--configs-file`` this reports the COCO metrics
(Box AP/AP50/AP75, and Mask AP/AP50/AP75 for Mask R-CNN) together with the
analytic backbone GMACs, then writes a summary table to the work dir.

Single GPU:
    python tools/eval_configs.py CONFIG CHECKPOINT --configs-file configs/eval_presets.yaml
Multi GPU:
    torchrun --nproc_per_node=8 tools/eval_configs.py CONFIG CHECKPOINT \\
        --configs-file configs/eval_presets.yaml --launcher pytorch
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import yaml
from mmengine.config import Config, DictAction
from mmengine.dist import is_main_process
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from elastic_vit_od.models.common import make_subnetwork_config  # noqa: E402
from elastic_vit_od.utils.flops import estimate_backbone_gmacs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Elastic-ViT OD over N backbone configs")
    parser.add_argument("config", help="Path to an MMDetection config")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--configs-file", required=True, help="YAML listing the N sub-networks to evaluate")
    parser.add_argument("--input-hw", type=int, nargs=2, default=(800, 1280), metavar=("H", "W"), help="Representative eval input size for the GMACs estimate")
    parser.add_argument("--work-dir", help="Directory to save the summary table")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="Override config options, key=value")
    return parser.parse_args()


def load_eval_configs(path: str, num_layers: int):
    with open(path, "r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    entries = spec["configs"] if isinstance(spec, dict) else spec
    parsed = []
    for index, entry in enumerate(entries):
        name = entry.get("name", f"config_{index}")
        if "mlp_widths" in entry or "num_heads_per_layer" in entry:
            mlp = entry["mlp_widths"]
            heads = entry["num_heads_per_layer"]
        else:
            mlp = entry["mlp_width"]
            heads = entry["num_heads"]
        config = make_subnetwork_config(mlp, heads, num_layers)
        parsed.append((name, config))
    return parsed


@torch.no_grad()
def evaluate_one(model, dataloader, evaluator) -> dict:
    model.eval()
    for data_batch in dataloader:
        outputs = model.test_step(data_batch)
        evaluator.process(data_samples=outputs, data_batch=data_batch)
    return evaluator.evaluate(len(dataloader.dataset))


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    cfg.load_from = None  # checkpoint loaded manually below
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    work_dir = Path(args.work_dir) if args.work_dir else Path("./work_dirs") / (Path(args.config).stem + "_eval_configs")
    cfg.work_dir = str(work_dir)

    runner = Runner.from_cfg(cfg)
    model = runner.model
    load_checkpoint(model, args.checkpoint, map_location="cpu", revise_keys=[(r"^module\.", "")])
    bare_model = model.module if is_model_wrapper(model) else model
    backbone = bare_model.backbone

    dataloader = Runner.build_dataloader(cfg.test_dataloader)
    dataset_meta = getattr(dataloader.dataset, "metainfo", None)

    eval_configs = load_eval_configs(args.configs_file, num_layers=backbone.num_layers)
    grid_h, grid_w = args.input_hw[0] // 16, args.input_hw[1] // 16

    rows = []
    for name, sub_config in eval_configs:
        backbone.set_active_config(sub_config.mlp_widths, sub_config.num_heads)
        evaluator = runner.build_evaluator(cfg.test_evaluator)
        evaluator.dataset_meta = dataset_meta
        metrics = evaluate_one(model, dataloader, evaluator)
        gmacs = estimate_backbone_gmacs(
            sub_config,
            embed_dim=backbone.embed_dim,
            grid_h=grid_h,
            grid_w=grid_w,
            num_prefix=backbone.num_prefix_tokens,
            max_heads=max(backbone.head_choices),
        )
        row = {"name": name, "backbone_gmacs": round(gmacs, 3), **{k: round(float(v), 4) for k, v in metrics.items()}}
        rows.append(row)
        if is_main_process():
            print(f"[{name}] GMACs={gmacs:.3f}  " + "  ".join(f"{k}={v}" for k, v in metrics.items()))

    if is_main_process() and rows:
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "eval_configs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        fieldnames = sorted({key for row in rows for key in row}, key=lambda k: (k != "name", k != "backbone_gmacs", k))
        with (work_dir / "eval_configs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved summary to {work_dir / 'eval_configs.json'} and .csv")


if __name__ == "__main__":
    main()
