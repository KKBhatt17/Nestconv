"""VQA model: elastic ViT vision tower + BLIP text/fusion + answer head.

Answering is framed as **classification over a fixed answer vocabulary** (see
``data/vocab.py``), which is what lets the curriculum trainer and the fixed-preset
evaluator carry over from the classification codebase almost unchanged.

Fusion reuses BLIP's multimodal text encoder: the question tokens are encoded and
**cross-attend to the elastic vision tokens** (passed as ``encoder_hidden_states``).
This is exactly how BLIP fuses modalities for VQA before its (here unused) answer
decoder. The pooled ``[CLS]`` of the fused sequence drives a linear answer head.

Only the elastic vision tower (and the answer head) train by default; the BLIP
text/fusion stack is frozen (``freeze_text=True``). Set ``freeze_text=False`` to
fine-tune it.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from elastic_vqa.models.common import SubnetworkConfig
from elastic_vqa.models.elastic_vit import ElasticVisionTransformer, RouterSoftConfig


class VqaElasticModel(nn.Module):
    def __init__(
        self,
        num_answers: int,
        vision_model_name: str = "vit_base_patch16_224",
        blip_model_name: str = "Salesforce/blip-vqa-base",
        vision_checkpoint: Optional[str] = None,
        vision_pretrained: bool = False,
        mlp_choices: Optional[list[int]] = None,
        head_choices: Optional[list[int]] = None,
        head_hidden_dim: int = 0,
        freeze_text: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vision = ElasticVisionTransformer(
            model_name=vision_model_name,
            pretrained=vision_pretrained,
            checkpoint_path=vision_checkpoint,
            mlp_choices=mlp_choices,
            head_choices=head_choices,
        )
        self.text_encoder = self._build_text_encoder(blip_model_name)
        self.freeze_text = freeze_text
        if freeze_text:
            for parameter in self.text_encoder.parameters():
                parameter.requires_grad = False

        embed_dim = self.vision.embed_dim
        text_dim = self._text_hidden_size()
        if text_dim != embed_dim:
            raise ValueError(
                f"Vision dim ({embed_dim}) must match the text encoder cross-attention dim "
                f"({text_dim}). Use a matching BLIP base + ViT-B pairing."
            )

        if head_hidden_dim and head_hidden_dim > 0:
            self.answer_head = nn.Sequential(
                nn.Linear(text_dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, num_answers),
            )
        else:
            self.answer_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(text_dim, num_answers))

        # Exposed for the FLOPs estimator and preset evaluator.
        self.embed_dim = embed_dim
        self.num_layers = self.vision.num_layers

    @staticmethod
    def _build_text_encoder(blip_model_name: str) -> nn.Module:
        from transformers import BlipForQuestionAnswering

        blip = BlipForQuestionAnswering.from_pretrained(blip_model_name)
        # text_encoder is BLIP's multimodal (cross-attention) encoder.
        return blip.text_encoder

    def _text_hidden_size(self) -> int:
        return int(self.text_encoder.config.hidden_size)

    def freeze_vision(self) -> None:
        self.vision.freeze()

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        config: Optional[SubnetworkConfig] = None,
        soft_config: Optional[RouterSoftConfig] = None,
    ) -> torch.Tensor:
        vision_tokens = self.vision(pixel_values, config=config, soft_config=soft_config)
        vision_atts = torch.ones(vision_tokens.shape[:-1], dtype=torch.long, device=vision_tokens.device)

        # Note: even when the text encoder is frozen we must NOT run it under
        # no_grad -- gradients still need to flow *through* it back to the vision
        # tower via cross-attention. Freezing its params (requires_grad=False) is
        # what keeps the text weights fixed.
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=vision_tokens,
            encoder_attention_mask=vision_atts,
            return_dict=True,
        )

        pooled = outputs.last_hidden_state[:, 0, :]
        return self.answer_head(pooled)
