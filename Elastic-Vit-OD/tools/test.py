"""Test an Elastic-ViT detector at the backbone config baked into the config file.

For evaluating several backbone sub-networks (with GMACs) in one go, use
``tools/eval_configs.py`` instead.

    python tools/test.py configs/mask_rcnn_elastic_vit_coco.py /path/to/checkpoint.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Elastic-ViT OD")
    parser.add_argument("config", help="Path to an MMDetection config")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--work-dir", help="Directory to save logs and predictions")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="Override config options, key=value")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    cfg.load_from = args.checkpoint

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = str(Path("./work_dirs") / Path(args.config).stem)

    runner = RUNNERS.build(cfg) if "runner_type" in cfg else Runner.from_cfg(cfg)
    runner.test()


if __name__ == "__main__":
    main()
