from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


DEFAULT_VGGT_REVISION = "860abec7937da0a4c03c41d3c269c366e82abdf9"


class VGGTEncoder(nn.Module):
    """Frozen VGGT aggregator that returns geometry-aware patch tokens."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        revision: str = DEFAULT_VGGT_REVISION,
        image_size: int = 518,
        initialize_from_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        from vggt.models.aggregator import Aggregator
        from vggt.models.vggt import VGGT

        if initialize_from_checkpoint:
            self.aggregator = Aggregator(img_size=image_size)
        else:
            loaded_model = VGGT.from_pretrained(model_name_or_path, revision=revision)
            self.aggregator = loaded_model.aggregator

        self.image_size = image_size
        self.aggregator.eval().requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.aggregator.eval()
        return self

    def _preprocess(self, images: Sequence[Sequence[Image.Image]]) -> torch.Tensor:
        batch = []
        for sample_images in images:
            views = []
            for image in sample_images:
                image = image.convert("RGB")
                width, height = image.size
                scale = self.image_size / max(width, height)
                resized_width = round(width * scale / 14) * 14
                resized_height = round(height * scale / 14) * 14
                image = image.resize(
                    (resized_width, resized_height), Image.Resampling.BICUBIC
                )
                padded = Image.new("RGB", (self.image_size, self.image_size), (255, 255, 255))
                padded.paste(
                    image,
                    (
                        (self.image_size - resized_width) // 2,
                        (self.image_size - resized_height) // 2,
                    ),
                )
                array = np.asarray(padded, dtype=np.float32).copy() / 255.0
                views.append(torch.from_numpy(array).permute(2, 0, 1))
            batch.append(torch.stack(views))
        return torch.stack(batch)

    def forward(self, images: Sequence[Sequence[Image.Image]], device: torch.device) -> torch.Tensor:
        image_tensor = self._preprocess(images).to(device=device, non_blocking=True)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            aggregated_tokens, patch_start_idx = self.aggregator(image_tensor)

        final_tokens = aggregated_tokens[-1]
        return final_tokens[:, :, patch_start_idx:, :]


class GatedFusionLayer(nn.Module):
    """Semantic-conditioned adaptive gate from equations (2)-(5) of 3D-MIX."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.semantic_projection = nn.Linear(hidden_dim, hidden_dim)
        self.geometry_projection = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        geometry_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        working_dtype = self.gate.weight.dtype
        hidden = hidden_states.to(dtype=working_dtype)
        geometry = geometry_tokens.to(dtype=working_dtype)

        mask = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        semantic = (hidden * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)

        semantic = semantic.expand(-1, geometry.shape[1], -1)
        gate = torch.sigmoid(self.gate(torch.cat((semantic, geometry), dim=-1)))
        fused = gate * self.semantic_projection(semantic)
        fused = fused + (1.0 - gate) * self.geometry_projection(geometry)
        return torch.cat((hidden_states, fused.to(dtype=hidden_states.dtype)), dim=1)


class ThreeDMixBridge(nn.Module):
    """Layer-wise 3D-MIX bridge between Qwen hidden states and the Q-Former."""

    def __init__(
        self,
        *,
        num_layers: int,
        hidden_dim: int,
        vggt_dim: int,
        vggt_model_name_or_path: str,
        vggt_revision: str = DEFAULT_VGGT_REVISION,
        image_size: int = 518,
        initialize_vggt_from_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.vggt_encoder = VGGTEncoder(
            vggt_model_name_or_path,
            revision=vggt_revision,
            image_size=image_size,
            initialize_from_checkpoint=initialize_vggt_from_checkpoint,
        )
        self.geometry_projection = nn.Linear(vggt_dim, hidden_dim)
        self.fusion_layers = nn.ModuleList(
            GatedFusionLayer(hidden_dim) for _ in range(num_layers)
        )

    def forward(
        self,
        hidden_states: Sequence[torch.Tensor],
        images: Sequence[Sequence[Image.Image]],
        attention_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        geometry = self.vggt_encoder(images, device=hidden_states[0].device)
        geometry = geometry.flatten(1, 2)
        geometry = self.geometry_projection(
            geometry.to(dtype=self.geometry_projection.weight.dtype)
        )
        return [
            layer(layer_hidden, geometry, attention_mask=attention_mask)
            for layer, layer_hidden in zip(self.fusion_layers, hidden_states)
        ]


def get_three_d_mix(config, *, num_layers: int) -> ThreeDMixBridge:
    cfg = config.framework.three_d_mix
    return ThreeDMixBridge(
        num_layers=num_layers,
        hidden_dim=cfg.hidden_dim,
        vggt_dim=cfg.vggt_dim,
        vggt_model_name_or_path=cfg.vggt_model_name_or_path,
        vggt_revision=cfg.vggt_revision,
        image_size=cfg.image_size,
        initialize_vggt_from_checkpoint=cfg.initialize_vggt_from_checkpoint,
    )
