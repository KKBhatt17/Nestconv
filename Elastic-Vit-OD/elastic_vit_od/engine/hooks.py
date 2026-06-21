"""Training hook that drives elastic sub-network sampling.

Mirrors Elastic-ViT stage-2: one per-layer sub-network is sampled each training
iteration under a large-to-small curriculum, and applied to the backbone via
``set_active_config``. The RNG is seeded by ``base_seed + global_step`` so every
DDP rank samples the *identical* config without any cross-rank communication
(all ranks must run the same sub-network for gradients to be consistent).
"""

from __future__ import annotations

import random
from typing import Sequence

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS

from elastic_vit_od.models.common import (
    DEFAULT_HEAD_CHOICES,
    DEFAULT_MLP_CHOICES,
    get_unlocked_choice_count,
    sample_layerwise_subnetwork,
)


@HOOKS.register_module()
class ElasticConfigHook(Hook):
    """Sample and set the backbone sub-network before every training iteration."""

    priority = "NORMAL"

    def __init__(
        self,
        mlp_choices: Sequence[int] = DEFAULT_MLP_CHOICES,
        head_choices: Sequence[int] = DEFAULT_HEAD_CHOICES,
        unlock_fractions: Sequence[float] = (0.0, 0.25, 0.5, 0.75),
        base_seed: int = 0,
        always_largest: bool = False,
    ) -> None:
        self.mlp_choices = list(mlp_choices)
        self.head_choices = list(head_choices)
        self.unlock_fractions = list(unlock_fractions)
        self.base_seed = base_seed
        self.always_largest = always_largest

    @staticmethod
    def _backbone(runner):
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        return model.backbone

    def before_train_iter(self, runner, batch_idx: int, data_batch=None) -> None:
        backbone = self._backbone(runner)
        if self.always_largest:
            backbone.set_active_config(max(self.mlp_choices), max(self.head_choices))
            return

        max_iters = max(getattr(runner.train_loop, "max_iters", 0) or 0, 1)
        progress = min(runner.iter / max_iters, 1.0)
        unlocked = get_unlocked_choice_count(progress, self.unlock_fractions)

        rng = random.Random(self.base_seed + runner.iter)
        config = sample_layerwise_subnetwork(
            num_layers=backbone.num_layers,
            mlp_choices=self.mlp_choices,
            head_choices=self.head_choices,
            unlocked_count=unlocked,
            rng=rng,
        )
        backbone.set_active_config(config.mlp_widths, config.num_heads)
