"""Pi05/PaliGemma adapter for the RL Token encoder data path.

Strict implementation of the RL Token paper's z_{1:M} construction (Sec. IV-A,
Eq. 1) for π0.5 (PaliGemmaPi05). Companion to the Qwen-only ``vla_features.py``;
the pretrain trainer dispatches on framework type.

Paper definition (Sec. IV-A):
    z = f(s, ℓ; θvla)              # final-layer token embeddings from the VLA
    z_{1:M} = {z_1, ..., z_M}      # one per input token
    z_rl = g_φ([z_{1:M}, e_rl])    # encoder summarizes into one RL token
    L_ro reconstructs sg(z_{1:M})  # autoregressive recon objective (Eq. 2)

For π0.5 the VLA backbone is SigLIP-So400m + Gemma 2B (Fig. 2). The
"final-layer token embeddings" therefore mean Gemma's last hidden state
over the prefix [image_embeddings, language_embeddings]. This is exactly
what Pi05's own inference code computes in the first stage of action
generation (``PaliGemmaPi05._predict_action``, lines 770-784): build the
prefix embeddings → run the VLM language model with a joint attention
mask and position ids → use ``hidden_states[-1]``.

We mirror that pipeline here, then expose only the image-token slice
(Fig. 2 / footnote 1: "each task has a fixed language instruction, so
we drop language embeddings in this step").

Phase-2 RL (rollout/actor) needs the parallel ``vla_actions`` chunk that
on Pi05 comes out of the flow-matching head (not in-stream action tokens
like Qwen). That adapter is intentionally out of scope for this file.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


# SigLIP-So400m at 224x224 with patch 14 → (224/14)^2 = 256 image embeds per view.
# Fixed by PaliGemma's vision tower; if you swap base_vlm with a different patch
# size, update this constant.
_PALIGEMMA_IMG_TOKENS_PER_VIEW = 256

# Sentinel used for HF-style additive masks (-inf-equivalent in bfloat16/float32).
_NEG_INF_ADDITIVE = -2.3819763e38  # matches PaliGemmaPi05.py:774


def _build_prefix_attn_4d(prefix_pad_masks, prefix_att_masks):
    """Mirror of PaliGemmaPi05's prefix attention setup (lines 771-774).

    Returns an additive 4D mask suitable for ``Gemma.forward(attention_mask=...)``
    when ``_attn_implementation = "eager"``.
    """
    from AlphaBrain.model.modules.action_model.pi0_flow_matching_head.openpi_inference import (
        make_att_2d_masks,
    )
    prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_att_4d = prefix_att_2d[:, None, :, :]
    prefix_att_4d = torch.where(prefix_att_4d, 0.0, _NEG_INF_ADDITIVE)
    return prefix_att_4d


@torch.no_grad()
def get_vla_hidden_states_pi05(
    vla,
    batch_images: List,
    instructions: List[str],
    image_only: bool = True,
    drop_action_tokens: bool = True,  # accepted for API parity; no-op for Pi05
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Return Pi05's z_{1:M} (paper Eq. 1) for a batch.

    Drop-in shape-compatible replacement for
    ``RLT.vla_features.get_vla_hidden_states``.

    Pipeline (mirrors PaliGemmaPi05._predict_action lines 762-784):
        1. ``_prepare_prefix_paligemma`` → image+text prefix embeddings.
        2. Build joint 4D attention mask + position ids.
        3. Run VLM language model (Gemma 2B) over prefix with
           ``output_hidden_states=True``.
        4. Take ``hidden_states[-1]`` — these are the paper's "final-layer
           token embeddings produced by the pretrained VLA".
        5. Optionally restrict to image-token positions (footnote 1).

    Returns:
        last_hidden  : (B, L, H_vlm)        — Gemma final-layer hidden state
        encoder_mask : (B, L) int 0/1        — positions to feed RL encoder
        action_mask  : None                  — Pi05 has no action-token mask
    """
    # Normalize images to a list-of-views per sample so PaliGemma's
    # _prepare_prefix_paligemma sees a uniform shape.
    batch_images_normed = [
        list(imgs) if isinstance(imgs, (list, tuple)) else [imgs]
        for imgs in batch_images
    ]
    examples = [
        {"image": imgs, "lang": instr}
        for imgs, instr in zip(batch_images_normed, instructions)
    ]

    # ── Step 1: build prefix embeddings (image+text, pre-Gemma) ──
    prefix_embs, prefix_pad_masks, prefix_att_masks = vla._prepare_prefix_paligemma(examples)
    # prefix_embs:      (B, L, H_vlm)  float, on device
    # prefix_pad_masks: (B, L) bool — True at valid (non-pad) positions
    # prefix_att_masks: (B, L) bool — segment boundaries for autoregressive groups

    # ── Step 2: 4D mask + position ids (PaliGemmaPi05.py:771-774) ──
    prefix_att_4d = _build_prefix_attn_4d(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    # ── Step 3: VLM language model (Gemma 2B) forward over prefix ──
    vlm_lm = vla._get_vlm_language_model()
    # The custom 4D additive mask is incompatible with flash_attention_2.
    # PaliGemmaPi05 itself flips this to "eager" at inference (line 776);
    # do the same. (Idempotent if already set.)
    vlm_lm.config._attn_implementation = "eager"
    prefix_output = vlm_lm.forward(
        inputs_embeds=prefix_embs,
        attention_mask=prefix_att_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        use_cache=False,
    )
    # Gemma's final-layer hidden state — the paper's z_{1:M}.
    # Use ``last_hidden_state`` (always populated) rather than
    # ``hidden_states[-1]`` which only fills when output_hidden_states=True
    # is honored, and HF's GemmaModel sometimes returns None for that field
    # depending on config.
    last_hidden = prefix_output.last_hidden_state  # (B, L, H_vlm)

    B, L, _H = last_hidden.shape

    # ── Step 4: image-only mask (footnote 1) or full validity mask ──
    if image_only:
        # PaliGemma layout: [view1_img_embeds, view2_img_embeds, ..., text_embeds].
        # SigLIP gives a fixed 256 embeds per 224x224 view.
        num_views = len(batch_images_normed[0])
        n_img = min(num_views * _PALIGEMMA_IMG_TOKENS_PER_VIEW, L)
        encoder_mask = torch.zeros(B, L, dtype=torch.long, device=last_hidden.device)
        encoder_mask[:, :n_img] = 1
        # AND with valid mask so any view marked invalid via image_masks is dropped.
        encoder_mask = encoder_mask & prefix_pad_masks.long()
    else:
        encoder_mask = prefix_pad_masks.to(torch.long)

    return last_hidden, encoder_mask, None
