from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from AlphaBrain.model.framework.base_framework import BaseFramework
from AlphaBrain.model.modules.action_model.spike_action_model_multitimestep import (
    get_action_model,
    get_gruedit_model,
)
from AlphaBrain.model.modules.projector.qformer import get_layerwise_qformer
from AlphaBrain.model.modules.vlm import get_vlm_model
from AlphaBrain.model.tools import FRAMEWORK_REGISTRY
from AlphaBrain.training.trainer_utils.trainer_tools import resize_images


@FRAMEWORK_REGISTRY.register("NeuroVLA")
class NeuroVLA(BaseFramework):
    """
    NeuroVLA: Vision-Language-Action model for robotic manipulation.

    This model combines a vision-language model (Qwen-VL) with action prediction
    to generate robot actions from visual observations and language instructions.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config

        # Vision-language model for processing images and instructions
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # Q-Former for extracting action-relevant features from VLM hidden states
        self.layer_qformer = get_layerwise_qformer(config=self.config)

        qformer_cfg = self.config.framework.layer_qformer
        action_cfg = self.config.framework.action_model
        architecture_version = int(self.config.framework.get("architecture_version", 1))

        feature_dim = int(qformer_cfg.get("output_dim", qformer_cfg.get("ouptput_dim", 768)))
        if architecture_version >= 2:
            action_hidden_dim = int(action_cfg.get("hidden_size", feature_dim * 2))
            self.action_dim = int(action_cfg.get("action_dim", 7))
            self.state_dim = int(action_cfg.get("state_dim", self.action_dim + 1))
            edit_hidden_dim = int(action_cfg.get("edit_hidden_size", 256))
            self.action_horizon = int(action_cfg.get("action_horizon", qformer_cfg.num_query_tokens))
        else:
            # Version-1 checkpoint configs recorded values that the old implementation
            # ignored. Preserve the architecture those checkpoints actually contain.
            action_hidden_dim = feature_dim * 2
            self.action_dim = 7
            self.state_dim = 8
            edit_hidden_dim = 256
            self.action_horizon = int(qformer_cfg.num_query_tokens)

        self.action_model = get_action_model(
            input_dim=feature_dim,
            hidden_dim=action_hidden_dim,
            action_dim=self.action_dim,
        )

        # Edit model for refining actions based on robot states
        self.edit_model = get_gruedit_model(
            input_dim=feature_dim,
            hidden_dim=edit_hidden_dim,
            robot_state_dim=self.state_dim,
        )

        self.l1_loss = nn.L1Loss()
        self.norm_stats = norm_stats

    def _roll_forward_states(self, states: torch.Tensor, predicted_actions: torch.Tensor) -> torch.Tensor:
        """Build the state history used to condition the next predicted chunk."""
        if states.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state dimension {self.state_dim}, got {states.shape[-1]}")

        predicted_states = torch.zeros_like(states)
        copied_steps = min(states.shape[1], predicted_actions.shape[1])
        copied_dims = min(states.shape[-1], predicted_actions.shape[-1])
        predicted_states[:, :copied_steps, :copied_dims] = predicted_actions[:, :copied_steps, :copied_dims]

        # Preserve state-only channels, such as LIBERO's eighth gripper-state channel.
        if copied_dims < states.shape[-1]:
            predicted_states[:, :, copied_dims:] = states[:, :, copied_dims:]
        return predicted_states

    def _predict_action_chunks(
        self,
        action_latent_feature: torch.Tensor,
        states: torch.Tensor,
        action_horizon: int,
    ) -> torch.Tensor:
        """Predict enough fixed-width Q-Former chunks to cover an action horizon."""
        if action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}")

        chunk_size = int(self.layer_qformer.num_query_tokens)
        num_iterations = math.ceil(action_horizon / chunk_size)
        predicted_chunks = []
        current_states = states

        for _ in range(num_iterations):
            edit_action_feature = self.edit_model(action_latent_feature, current_states)
            predicted_actions = self.action_model.predict_action(edit_action_feature)
            if predicted_actions.shape[1] != chunk_size:
                raise ValueError(
                    f"Action head returned {predicted_actions.shape[1]} steps; expected Q-Former chunk size {chunk_size}"
                )
            predicted_chunks.append(predicted_actions)
            current_states = self._roll_forward_states(current_states, predicted_actions)

        return torch.cat(predicted_chunks, dim=1)[:, :action_horizon]

    def forward(
        self,
        examples: List[dict] = None,
        repeated_diffusion_steps: int = 4,
        **kwargs,
    ) -> Tuple:
        """
        Run a forward pass through the VLM, returning loss for training.

        Args:
            examples: List of training examples, each containing:
                - "image": Input images
                - "lang": Language instructions
                - "action": Ground truth actions [B, T, 7]
                - "state": Robot states [B, T, 8]
                - "solution" (optional): Chain-of-thought solutions

        Returns:
            Dictionary containing action_loss
        """
        # Extract data from examples
        images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        assert "state" in examples[0], (
            "NeuroVLA requires 'state' in training samples. "
            "Please set 'include_state: true' in your dataset config yaml (datasets.vla_data.include_state)."
        )
        states = [example["state"] for example in examples]

        if "solution" in examples[0]:
            solutions = [example["solution"] for example in examples]
        else:
            solutions = None

        # Build inputs for vision-language model
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=images, instructions=instructions, solutions=solutions
        )

        # Forward pass through VLM to get hidden states
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

        vlm_cot_loss = qwenvl_outputs.loss

        if vlm_cot_loss is None or torch.isnan(vlm_cot_loss):
            vlm_cot_loss = torch.tensor(0.0, device=self.qwen_vl_interface.model.device)

        # Action prediction with iterative refinement
        with torch.autocast("cuda", dtype=torch.float32):
            # Extract action-relevant features from VLM hidden states
            start_layer = self.config.framework.layer_qformer.qformer_start_layer if self.config else -6
            end_layer = self.config.framework.layer_qformer.qformer_end_layer if self.config else -1
            action_latent_feature = self.layer_qformer(qwenvl_outputs.hidden_states[start_layer:end_layer])

            states = torch.tensor(np.array(states), dtype=torch.float32, device=action_latent_feature.device)
            action_horizon = np.array(actions).shape[1]  # total action steps from ground truth
            predicted_action_tensor = self._predict_action_chunks(action_latent_feature, states, action_horizon)
            action_tensor = torch.tensor(np.array(actions), dtype=torch.float32, device=predicted_action_tensor.device)
            if action_tensor.shape[-1] != self.action_dim:
                raise ValueError(f"Expected action dimension {self.action_dim}, got {action_tensor.shape[-1]}")
            action_loss = self.l1_loss(predicted_action_tensor, action_tensor)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: Union[Image, List[Image]],
        instructions: List[str],
        states: Optional[List[Sequence[float]]] = None,
        solutions: Union[Dict, List[Dict]] = None,
        unnorm_key: Optional[str] = None,
        cfg_scale: float = 1.5,
        use_ddim: bool = False,
        num_ddim_steps: int = 5,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Predict action from images and instructions.

        Args:
            batch_images: Input images (PIL Image or list of PIL Images)
            instructions: Task instructions (list of strings)
            states: Robot states history [B, T, 8], where last dim is [x,y,z,roll,pitch,yaw,gripper,pad]
            solutions: Optional solution dict for chain-of-thought
            unnorm_key: Key for unnormalization (if using norm_stats)
            cfg_scale: Classifier-free guidance scale (>1.0 enables CFG)
            use_ddim: Whether to use DDIM sampling
            num_ddim_steps: Number of DDIM steps

        Returns:
            Dictionary containing "normalized_actions" [B, T, 7]
        """
        # ! [zhanghe] 将client端的array转化为PIL; 后续考虑在其他地方处理；
        if isinstance(batch_images[0][0], np.ndarray):
            batch_images = [[Image.fromarray(img) for img in seq] for seq in batch_images]

        batch_images = resize_images(batch_images, target_size=(224, 224))

        # Build VLM inputs
        interface_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        qwen_inputs = interface_inputs

        # Generate cognition features through VLM
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                input_ids=qwen_inputs.input_ids,
                attention_mask=qwen_inputs.attention_mask,
                pixel_values=qwen_inputs.pixel_values,
                image_grid_thw=qwen_inputs.image_grid_thw,
                labels=qwen_inputs.input_ids.clone(),
                output_hidden_states=True,
                return_dict=True,
            )

        # Action prediction with iterative refinement
        with torch.autocast("cuda", dtype=torch.float32):
            # Extract action features from VLM hidden states
            start_layer = self.config.framework.layer_qformer.qformer_start_layer if self.config else -2
            end_layer = self.config.framework.layer_qformer.qformer_end_layer if self.config else -1

            action_latent_feature = self.layer_qformer(qwenvl_outputs.hidden_states[start_layer:end_layer])

            # Convert states to tensor
            if states is None:
                raise ValueError("NeuroVLA requires robot states for action prediction")
            states = torch.tensor(
                np.array(states, dtype=np.float32),
                dtype=torch.float32,
                device=action_latent_feature.device
            )

            predicted_action_tensor = self._predict_action_chunks(
                action_latent_feature,
                states,
                self.action_horizon,
            )

        normalized_actions = predicted_action_tensor.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


def build_model_framework(config: dict = {}) -> NeuroVLA:
    """Build NeuroVLA model from config."""
    model = NeuroVLA(config=config)
    return model


if __name__ == "__main__":
    """
    Example usage for testing the model.

    This demonstrates how to:
    1. Load a pretrained model
    2. Prepare input data
    3. Run inference to predict actions
    """
    import pickle
    from omegaconf import OmegaConf

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Option 1: Load from pretrained checkpoint
    # model = NeuroVLA.from_pretrained("path/to/checkpoint.pt").to(device)
    model = NeuroVLA.from_pretrained("/workspace/nature_submit/NeuroVLA/data/checkpoints/1104_neurovla_gru_xiaonao_goal_dualimage_spike_multistep_ac8_768*2_yibu/checkpoints/steps_10000_pytorch_model.pt").to(device)
    # Option 2: Build from config
    # config = OmegaConf.load("path/to/config.yaml")
    # model = NeuroVLA(config).to(device)

    # Prepare sample data
    # Each sample should contain:
    # - "image": List of PIL Images
    # - "lang": Language instruction (string)
    # - "state": Robot state history [T, 8]
    # - "action": Ground truth actions [T, 7] (for training only)

    # Example data structure:
    # samples = [
    #     {
    #         "image": [],  # List of PIL Images
    #         "lang": "pick up the red block",
    #         "state": np.zeros((16, 8)),  # [T, 8] state history
    #         "action": np.zeros((8, 7)),  # [T, 7] action sequence
    #     }
    # ]
    import pickle
    from omegaconf import OmegaConf
    with open("/workspace/samples_states.pkl", "rb") as f:
        samples = pickle.load(f)
    device = torch.device("cuda:0")

    # Extract data for inference
    images = [sample["image"] for sample in samples]
    instructions = [sample["lang"] for sample in samples]
    states = [sample["state"] for sample in samples]

    # Run inference
    with torch.inference_mode():
        result = model.predict_action(
            batch_images=images,
            instructions=instructions,
            states=states,
        )
        normalized_actions = result["normalized_actions"]
        print(f"Predicted actions shape: {normalized_actions.shape}")

    print("Test example ready. Uncomment the code above to run inference.")
