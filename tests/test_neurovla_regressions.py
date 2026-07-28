import os
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from AlphaBrain.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from AlphaBrain.model.framework.NeuroVLA import NeuroVLA
from AlphaBrain.model.modules.projector.qformer import LayerwiseQFormer
from AlphaBrain.training.trainer_utils.finetune_config import build_config_from_finetune


class DummyQFormer(nn.Module):
    def __init__(self, num_query_tokens):
        super().__init__()
        self.num_query_tokens = num_query_tokens


def make_config(version=2):
    framework = {
        "name": "NeuroVLA",
        "layer_qformer": {
            "qformer_start_layer": 0,
            "qformer_end_layer": 1,
            "num_query_tokens": 4,
            "input_dim": 5,
            "output_dim": 3,
            "grad_scale": 0.5,
        },
        "action_model": {
            "hidden_size": 6,
            "edit_hidden_size": 4,
            "action_dim": 2,
            "state_dim": 3,
            "action_horizon": 6,
        },
    }
    if version is not None:
        framework["architecture_version"] = version
    return OmegaConf.create({"framework": framework})


class NeuroVLARegressionTests(unittest.TestCase):
    def build_without_vlm(self, config):
        qformer = DummyQFormer(config.framework.layer_qformer.num_query_tokens)
        with (
            patch("AlphaBrain.model.framework.NeuroVLA.get_vlm_model", return_value=nn.Identity()),
            patch("AlphaBrain.model.framework.NeuroVLA.get_layerwise_qformer", return_value=qformer),
        ):
            return NeuroVLA(config)

    def test_versioned_dimensions_and_arbitrary_horizon(self):
        model = self.build_without_vlm(make_config())

        self.assertEqual(model.action_model.model.fc1.in_features, 3)
        self.assertEqual(model.action_model.model.fc1.out_features, 6)
        self.assertEqual(model.action_model.model.fc3.out_features, 2)
        self.assertEqual(model.edit_model.robot_state_encoder.input_size, 3)
        self.assertEqual(model.action_horizon, 6)

        latent = torch.randn(2, 4, 3, requires_grad=True)
        states = torch.randn(2, 5, 3)
        actions = model._predict_action_chunks(latent, states, action_horizon=6)
        self.assertEqual(tuple(actions.shape), (2, 6, 2))
        actions.sum().backward()
        self.assertIsNotNone(latent.grad)

    def test_unversioned_checkpoint_config_uses_legacy_dimensions(self):
        config = make_config(version=None)
        config.framework.layer_qformer.ouptput_dim = config.framework.layer_qformer.pop("output_dim")
        config.framework.action_model.hidden_size = 99
        config.framework.action_model.action_dim = 9
        config.framework.action_model.state_dim = 9
        config.framework.action_model.action_horizon = 9

        model = self.build_without_vlm(config)
        self.assertEqual(model.action_model.model.fc1.out_features, 6)
        self.assertEqual(model.action_model.model.fc3.out_features, 7)
        self.assertEqual(model.edit_model.robot_state_encoder.input_size, 8)
        self.assertEqual(model.action_horizon, 4)

    def test_qformer_grad_scale_preserves_values_and_scales_gradients(self):
        config = make_config()
        qformer = LayerwiseQFormer(
            input_hidden_dim=3,
            output_hidden_dim=3,
            num_query_tokens=2,
            num_layers=1,
            num_heads=1,
            config=config,
        )
        hidden_state = torch.randn(2, 4, 3, requires_grad=True)
        scaled = qformer.scale_hook([hidden_state])[0]

        torch.testing.assert_close(scaled, hidden_state)
        scaled.sum().backward()
        torch.testing.assert_close(hidden_state.grad, torch.full_like(hidden_state, 0.5))

    def test_neurovla_mode_uses_lerobot_root(self):
        with patch.dict(os.environ, {"LEROBOT_LIBERO_DATA_DIR": "/datasets/lerobot"}):
            config = build_config_from_finetune(
                OmegaConf.load("configs/finetune_config_ga2.yaml"),
                "neuro_vla",
            )
        self.assertEqual(config.datasets.vla_data.data_root_dir, "/datasets/lerobot")

    def test_libero_10_alias(self):
        self.assertEqual(DATASET_NAMED_MIXTURES["libero_10"], DATASET_NAMED_MIXTURES["libero_long"])


if __name__ == "__main__":
    unittest.main()
