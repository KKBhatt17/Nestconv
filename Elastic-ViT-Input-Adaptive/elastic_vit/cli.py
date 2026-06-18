from __future__ import annotations

import argparse

from elastic_vit.config import load_config
from elastic_vit.engine.rearrange import run_rearrangement
from elastic_vit.engine.train_elastic import run_elastic_training
from elastic_vit.engine.train_router import run_router_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Elastic Nested ViT pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("rearrange", "train-elastic", "train-router"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True, help="Path to YAML config")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "rearrange":
        run_rearrangement(config)
        return
    if args.command == "train-elastic":
        run_elastic_training(config)
        return
    if args.command == "train-router":
        run_router_training(config)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
