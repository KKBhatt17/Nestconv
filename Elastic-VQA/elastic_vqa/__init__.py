"""Elastic Nested ViT for Visual Question Answering.

A two-stage pipeline (rearrange + curriculum elastic training) that turns a
pretrained ViT-B/16 vision tower into a width-elastic backbone and trains a VQA
model on top of it. Adapted from the Elastic-ViT classification codebase, but
the router stage and entropy-sorted loading are intentionally dropped: VQA here
is curriculum-trained and evaluated at a fixed set of backbone presets.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
