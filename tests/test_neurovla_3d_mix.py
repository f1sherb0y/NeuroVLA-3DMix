import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image

from AlphaBrain.model.framework.NeuroVLA3DMix import NeuroVLA3DMix
from AlphaBrain.model.modules.projector.qformer import LayerwiseQFormer
from AlphaBrain.model.modules.projector.three_d_mix import (
    GatedFusionLayer,
    VGGTEncoder,
)
from AlphaBrain.training.trainer_utils.finetune_config import build_config_from_finetune


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class NeuroVLA3DMixTests(unittest.TestCase):
    def test_checkpoint_skeleton_does_not_read_external_vggt_weights(self):
        class FakeAggregator(nn.Module):
            def __init__(self, img_size):
                super().__init__()
                self.img_size = img_size
                self.weight = nn.Parameter(torch.ones(1))

        class FakeVGGT:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise AssertionError("external VGGT weights must not be read")

        vggt_package = types.ModuleType("vggt")
        models_package = types.ModuleType("vggt.models")
        aggregator_module = types.ModuleType("vggt.models.aggregator")
        model_module = types.ModuleType("vggt.models.vggt")
        aggregator_module.Aggregator = FakeAggregator
        model_module.VGGT = FakeVGGT

        with patch.dict(
            sys.modules,
            {
                "vggt": vggt_package,
                "vggt.models": models_package,
                "vggt.models.aggregator": aggregator_module,
                "vggt.models.vggt": model_module,
            },
        ):
            encoder = VGGTEncoder(
                "/missing/VGGT-1B",
                initialize_from_checkpoint=True,
            )

        self.assertEqual(encoder.aggregator.img_size, 518)
        self.assertFalse(encoder.aggregator.weight.requires_grad)

    def test_vggt_preprocessing_keeps_batched_multiview_layout(self):
        encoder = object.__new__(VGGTEncoder)
        nn.Module.__init__(encoder)
        encoder.image_size = 518
        images = [[Image.new("RGB", (320, 160), "black"), Image.new("RGB", (256, 256))]]

        tensor = encoder._preprocess(images)

        self.assertEqual(tensor.shape, (1, 2, 3, 518, 518))
        self.assertEqual(tensor.dtype, torch.float32)
        torch.testing.assert_close(tensor[0, 0, :, 0, 0], torch.ones(3))
        torch.testing.assert_close(tensor[0, 0, :, 259, 259], torch.zeros(3))

    def test_gated_fusion_matches_paper_equation(self):
        layer = GatedFusionLayer(hidden_dim=2)
        with torch.no_grad():
            layer.semantic_projection.weight.copy_(torch.eye(2))
            layer.semantic_projection.bias.zero_()
            layer.geometry_projection.weight.copy_(torch.eye(2))
            layer.geometry_projection.bias.zero_()
            layer.gate.weight.zero_()
            layer.gate.bias.zero_()

        hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
        geometry = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])
        output = layer(hidden, geometry, attention_mask=torch.tensor([[1, 1, 0]]))

        semantic = torch.tensor([2.0, 4.0])
        expected_fused = 0.5 * semantic + 0.5 * geometry[0]
        torch.testing.assert_close(output[:, : hidden.shape[1]], hidden)
        torch.testing.assert_close(output[0, hidden.shape[1] :], expected_fused)

    def test_training_uses_original_resolution_geometry_images(self):
        model = object.__new__(NeuroVLA3DMix)
        geometry_images = [[object(), object()]]
        result = model._geometry_images_from_examples(
            [{"image": [object()], "vggt_image": geometry_images[0]}],
            [[object()]],
        )
        self.assertIs(result[0][0], geometry_images[0][0])

    def test_qformer_can_skip_interface_gradient_scaling(self):
        class Config:
            class framework:
                class layer_qformer:
                    grad_scale = 0.5

        qformer = LayerwiseQFormer(
            input_hidden_dim=2,
            output_hidden_dim=2,
            num_query_tokens=1,
            num_layers=1,
            num_heads=1,
            config=Config(),
        ).eval()
        hidden_scaled = torch.ones(1, 2, 2, requires_grad=True)
        hidden_unscaled = hidden_scaled.detach().clone().requires_grad_(True)

        qformer([hidden_scaled]).sum().backward()
        qformer([hidden_unscaled], apply_grad_scale=False).sum().backward()

        torch.testing.assert_close(hidden_scaled.grad * 2.0, hidden_unscaled.grad)

    def test_finetune_modes_match_baseline_schedule(self):
        with patch.dict(os.environ, {"VGGT_MODEL_PATH": "/models/VGGT-1B"}):
            for config_name, expected_batch, expected_accumulation in (
                ("finetune_config.yaml", 16, 1),
                ("finetune_config_ga2.yaml", 8, 2),
            ):
                config = OmegaConf.load(PROJECT_ROOT / "configs" / config_name)
                baseline = build_config_from_finetune(config, "neuro_vla")
                three_d_mix = build_config_from_finetune(config, "neuro_vla_3d_mix")

                self.assertEqual(three_d_mix.framework.name, "NeuroVLA3DMix")
                self.assertEqual(
                    three_d_mix.framework.three_d_mix.vggt_model_name_or_path,
                    "/models/VGGT-1B",
                )
                self.assertTrue(three_d_mix.datasets.vla_data.include_vggt_images)
                self.assertEqual(three_d_mix.datasets.vla_data.per_device_batch_size, expected_batch)
                self.assertEqual(three_d_mix.trainer.gradient_accumulation_steps, expected_accumulation)
                self.assertEqual(
                    three_d_mix.trainer.max_train_steps,
                    baseline.trainer.max_train_steps,
                )
                self.assertEqual(three_d_mix.trainer.learning_rate.three_d_mix, 1.0e-4)


if __name__ == "__main__":
    unittest.main()
