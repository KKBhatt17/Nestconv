from __future__ import annotations

import argparse
from pathlib import Path
import sys

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmseg.utils import register_all_modules

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Elastic-ViT SS with pip-installed MMSegmentation")
    parser.add_argument("config", help="Path to an MMSegmentation config")
    parser.add_argument("--work-dir", help="Directory to save logs and checkpoints")
    parser.add_argument("--resume", nargs="?", const="auto", help="Resume from a checkpoint path, or auto if no path is given")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="Override config options, key=value")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_all_modules(init_default_scope=False)
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = str(Path("./work_dirs") / Path(args.config).stem)

    if args.amp:
        cfg.optim_wrapper.type = "AmpOptimWrapper"
        cfg.optim_wrapper.setdefault("loss_scale", "dynamic")

    if args.resume == "auto":
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    runner = RUNNERS.build(cfg) if "runner_type" in cfg else Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
