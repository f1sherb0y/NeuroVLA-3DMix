from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


DEFAULT_VGGT_REVISION = "860abec7937da0a4c03c41d3c269c366e82abdf9"


class VGGTEncoder(nn.Module):
    """Frozen VGGT aggregator that returns geometry-aware patch tokens."""

    def __init__(
        self,
        model_name_or_path: Optional[str],
        *,
        revision: str = DEFAULT_VGGT_REVISION,
        image_size: int = 518,
        initialize_from_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if image_size <= 0 or image_size % 14 != 0:
            raise ValueError(f"VGGT image_size must be positive and divisible by 14, got {image_size}")

        try:
            from vggt.models.aggregator import Aggregator
            from vggt.models.vggt import VGGT
        except ImportError as exc:
            raise ImportError(
                "NeuroVLA3DMix requires the pinned VGGT package. "
                "Run: bash scripts/setup_neurovla_env.sh"
            ) from exc

        if initialize_from_checkpoint:
            # BaseFramework.from_pretrained fills this randomly initialized
            # skeleton from the self-contained NeuroVLA3DMix state dict.
            self.aggregator = Aggregator(img_size=image_size)
        else:
            if not model_name_or_path:
                raise ValueError(
                    "framework.three_d_mix.vggt_model_name_or_path is required "
                    "when starting NeuroVLA3DMix training"
                )
            expanded_path = Path(model_name_or_path).expanduser()
            path_like = model_name_or_path.startswith(("/", ".", "~"))
            if path_like and not expanded_path.exists():
                raise FileNotFoundError(f"VGGT checkpoint path does not exist: {expanded_path}")
            resolved_path = str(expanded_path) if path_like else model_name_or_path
            loaded_model = VGGT.from_pretrained(resolved_path, revision=revision)
            self.aggregator = loaded_model.aggregator

        self.image_size = image_size
        self.aggregator.eval().requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.aggregator.eval()
        return self

    def _preprocess(self, images: Sequence[Sequence[Image.Image]]) -> torch.Tensor:
        if not images or not images[0]:
            raise ValueError("VGGT requires at least one image per batch item")

        expected_views = len(images[0])
        batch = []
        for sample_index, sample_images in enumerate(images):
            if len(sample_images) != expected_views:
                raise ValueError(
                    "All NeuroVLA3DMix samples must have the same number of views; "
                    f"sample 0 has {expected_views}, sample {sample_index} has {len(sample_images)}"
                )
            views = []
            for image in sample_images:
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                if not isinstance(image, Image.Image):
                    raise TypeError(f"Expected PIL image or ndarray, got {type(image)!r}")
                image = image.convert("RGB")
                width, height = image.size
                if width <= 0 or height <= 0:
                    raise ValueError(f"VGGT received an empty image with size {image.size}")

                # Match VGGT's official `mode="pad"` preprocessing: preserve
                # aspect ratio, make the resized dimensions divisible by the
                # 14-pixel patch size, and center-pad with white.
                scale = self.image_size / max(width, height)
                resized_width = max(14, round(width * scale / 14) * 14)
                resized_height = max(14, round(height * scale / 14) * 14)
                resized_width = min(self.image_size, resized_width)
                resized_height = min(self.image_size, resized_height)
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
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with torch.no_grad(), autocast:
            aggregated_tokens, patch_start_idx = self.aggregator(image_tensor)

        final_tokens = aggregated_tokens[-1]
        if final_tokens is None:
            raise RuntimeError("VGGT did not cache its final aggregator layer")
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
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        working_dtype = self.gate.weight.dtype
        hidden = hidden_states.to(dtype=working_dtype)
        geometry = geometry_tokens.to(dtype=working_dtype)

        if attention_mask is None:
            semantic = hidden.mean(dim=1, keepdim=True)
        else:
            mask = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
            semantic = (hidden * mask).sum(dim=1, keepdim=True) / mask.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)

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
        vggt_model_name_or_path: Optional[str],
        vggt_revision: str = DEFAULT_VGGT_REVISION,
        image_size: int = 518,
        max_geometry_tokens: int = 0,
        initialize_vggt_from_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if max_geometry_tokens < 0:
            raise ValueError(f"max_geometry_tokens cannot be negative, got {max_geometry_tokens}")

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
        self.max_geometry_tokens = max_geometry_tokens

    def _pool_geometry_tokens(self, geometry_tokens: torch.Tensor) -> torch.Tensor:
        # Preserve view boundaries while reducing each square VGGT patch grid.
        batch_size, num_views, patches_per_view, channels = geometry_tokens.shape
        if self.max_geometry_tokens == 0 or num_views * patches_per_view <= self.max_geometry_tokens:
            return geometry_tokens.flatten(1, 2)

        source_grid = int(patches_per_view**0.5)
        if source_grid * source_grid != patches_per_view:
            raise ValueError(f"Expected a square VGGT patch grid, got {patches_per_view} tokens/view")

        target_per_view = max(1, self.max_geometry_tokens // num_views)
        target_grid = max(1, int(target_per_view**0.5))
        tokens = geometry_tokens.reshape(
            batch_size * num_views, source_grid, source_grid, channels
        ).permute(0, 3, 1, 2)
        tokens = F.adaptive_avg_pool2d(tokens, (target_grid, target_grid))
        return tokens.permute(0, 2, 3, 1).reshape(batch_size, -1, channels)

    def forward(
        self,
        hidden_states: Sequence[torch.Tensor],
        images: Sequence[Sequence[Image.Image]],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        if len(hidden_states) != len(self.fusion_layers):
            raise ValueError(
                f"3D-MIX received {len(hidden_states)} hidden layers, "
                f"expected {len(self.fusion_layers)}"
            )

        geometry = self.vggt_encoder(images, device=hidden_states[0].device)
        geometry = self._pool_geometry_tokens(geometry)
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
        hidden_dim=int(cfg.hidden_dim),
        vggt_dim=int(cfg.vggt_dim),
        vggt_model_name_or_path=cfg.get("vggt_model_name_or_path"),
        vggt_revision=str(cfg.get("vggt_revision", DEFAULT_VGGT_REVISION)),
        image_size=int(cfg.get("image_size", 518)),
        max_geometry_tokens=int(cfg.get("max_geometry_tokens", 0)),
        initialize_vggt_from_checkpoint=bool(cfg.get("initialize_vggt_from_checkpoint", False)),
    )
