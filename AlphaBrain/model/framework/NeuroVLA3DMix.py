from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from PIL import Image

from AlphaBrain.model.framework.NeuroVLA import NeuroVLA
from AlphaBrain.model.modules.projector.three_d_mix import get_three_d_mix
from AlphaBrain.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("NeuroVLA3DMix")
class NeuroVLA3DMix(NeuroVLA):
    """NeuroVLA with frozen VGGT geometry and layer-wise 3D-MIX fusion."""

    def __init__(self, config=None, norm_stats=None, **kwargs) -> None:
        super().__init__(config=config, norm_stats=norm_stats, **kwargs)
        self.three_d_mix = get_three_d_mix(
            config=self.config,
            num_layers=self.layer_qformer.num_layers,
        )

    def _geometry_images_from_examples(
        self,
        examples: List[dict],
        images: List[List[Image.Image]],
    ) -> List[List[Image.Image]]:
        return [example["vggt_image"] for example in examples]

    def _prepare_action_hidden_states(
        self,
        hidden_states: Sequence[torch.Tensor],
        geometry_images: List[List[Image.Image]],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Sequence[torch.Tensor]:
        return self.three_d_mix(
            hidden_states,
            geometry_images,
            attention_mask=attention_mask,
        )

    def _encode_action_features(
        self,
        hidden_states: Sequence[torch.Tensor],
        geometry_images: List[List[Image.Image]],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.layer_qformer.scale_hook(hidden_states)
        hidden_states = self._prepare_action_hidden_states(
            hidden_states,
            geometry_images,
            attention_mask=attention_mask,
        )
        geometry_mask = attention_mask.new_ones(
            attention_mask.shape[0],
            hidden_states[0].shape[1] - attention_mask.shape[1],
        )
        conditioning_mask = torch.cat((attention_mask, geometry_mask), dim=1)
        return self.layer_qformer(
            hidden_states,
            encoder_attention_mask=conditioning_mask,
            apply_grad_scale=False,
        )


def build_model_framework(config: dict = {}) -> NeuroVLA3DMix:
    return NeuroVLA3DMix(config=config)
