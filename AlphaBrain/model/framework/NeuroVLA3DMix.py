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
        missing = [index for index, example in enumerate(examples) if "vggt_image" not in example]
        if missing:
            raise ValueError(
                "NeuroVLA3DMix training requires original-resolution `vggt_image` data. "
                "Set datasets.vla_data.include_vggt_images: true. "
                f"Missing samples: {missing[:5]}"
            )
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
        # NeuroVLA's grad_scale belongs to the Qwen-to-action interface. Apply
        # it before fusion so it does not also attenuate 3D-MIX gradients.
        hidden_states = self.layer_qformer.scale_hook(hidden_states)
        hidden_states = self._prepare_action_hidden_states(
            hidden_states,
            geometry_images,
            attention_mask=attention_mask,
        )
        return self.layer_qformer(hidden_states, apply_grad_scale=False)


def build_model_framework(config: dict = {}) -> NeuroVLA3DMix:
    return NeuroVLA3DMix(config=config)
