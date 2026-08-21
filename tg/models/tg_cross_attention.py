# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sentence-tokens -> STM cross-attention.

Port of `models/tg/cross_attention.py:CrossAttentionDispatcher` (plus the
`memory_dim == d_model` branch of `models/tg/attention_core.py:SDPAAttention`).

Two behaviours are load-bearing and copied exactly:
  * the sinusoidal positional encoding is L2-normalized and added to the
    cross-attention KEYS only -- the VALUES are the raw S_REPs;
  * rows whose STM holds no entries get an exactly-zero cross-attention output
    (the reference achieves this by `index_select`-ing the rows that do have
    memory and scattering zeros elsewhere).
"""

# pylint: disable=invalid-name,g-importing-member

from functools import partial

from typing import Optional

from flax import linen as nn
import jax
import jax.numpy as jnp
from tg.models.tg_config import TgConfig
from tg.models.tg_config import tg_init


def grouped_memory_positions(
    mem_valid_BxM: jax.Array, mem_step_BxM: jax.Array
) -> jax.Array:
  """Positions for `multi_eos_pos_mode="grouped"`: one index per SENTENCE.

  Each distinct source sentence step takes the next integer in order of first
  appearance, so the K vectors from one sentence share a position instead of
  occupying K consecutive ones. Steps are non-decreasing along our oldest-first
  prefix, so that rank is a running count of group changes. Empty slots: -1.
  """
  prev_step_BxM = jnp.concatenate(
      [jnp.full_like(mem_step_BxM[:, :1], -1), mem_step_BxM[:, :-1]], axis=1
  )
  prev_valid_BxM = jnp.concatenate(
      [jnp.zeros_like(mem_valid_BxM[:, :1]), mem_valid_BxM[:, :-1]], axis=1
  )
  # A slot starts a new group when it is the first valid slot, or its step
  # differs from the previous valid slot's.
  starts_BxM = mem_valid_BxM & (
      (~prev_valid_BxM) | (mem_step_BxM != prev_step_BxM)
  )
  rank_BxM = jnp.cumsum(starts_BxM.astype(jnp.int32), axis=1) - 1
  return jnp.where(mem_valid_BxM, rank_BxM, -1)


def memory_positions(mem_valid_BxM: jax.Array) -> jax.Array:
  """Position of each STM slot, oldest -> newest; -1 for empty slots.

  `_build_stm_tensor` stores the deque reversed (oldest first) and labels the
  `K` occupied slots `arange(K)`, padding the rest with -1. Our STM keeps
  occupied slots in a prefix, so `arange(M)` masked by validity is identical.
  """
  M = mem_valid_BxM.shape[-1]
  idx_1xM = jnp.arange(M, dtype=jnp.int32)[None, :]
  return jnp.where(mem_valid_BxM, idx_1xM, -1)


def sinusoidal_key_pe(
    positions_BxM: jax.Array,
    D: int,
    positional_weight: float = 1.0,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
  """L2-normalized sinusoidal encoding for STM keys.

  Mirrors `CrossAttentionDispatcher._memory_positional_encoding` for
  `positional_mode == "sinusoidal"`: even dims sin, odd dims cos, the whole
  vector is L2-normalized (eps 1e-6), zeroed on invalid slots, then scaled by
  `positional_weight`.
  """
  if D % 2 != 0:
    raise ValueError('memory_dim must be even for sinusoidal STM positions.')
  valid_BxM = positions_BxM >= 0
  # Computed in float32 regardless of `dtype`; cast at the end.
  inv_freq_Dh = jnp.exp(
      jnp.arange(0, D, 2, dtype=jnp.float32) * -(jnp.log(10000.0) / D)
  )
  clamped_BxM = jnp.maximum(positions_BxM, 0).astype(jnp.float32)
  angles_BxMxDh = clamped_BxM[..., None] * inv_freq_Dh[None, None, :]
  pe_BxMxDhx2 = jnp.stack(
      [jnp.sin(angles_BxMxDh), jnp.cos(angles_BxMxDh)], axis=-1
  )
  pe_BxMxD = pe_BxMxDhx2.reshape(*positions_BxM.shape, D)
  norm_BxMx1 = jnp.maximum(
      jnp.linalg.norm(pe_BxMxD, axis=-1, keepdims=True), 1e-6
  )
  pe_BxMxD = pe_BxMxD / norm_BxMx1
  pe_BxMxD = pe_BxMxD * valid_BxM[..., None].astype(pe_BxMxD.dtype)
  return (pe_BxMxD * positional_weight).astype(dtype)


class TgCrossAttn(nn.Module):
  """Cross-attention from sentence tokens (queries) to STM (keys/values).

  Non-causal: every token may look at every occupied memory slot.
  """

  cfg: TgConfig

  @nn.compact
  def __call__(
      self,
      x_BxLxD: jax.Array,
      mem_BxMxD: jax.Array,
      mem_valid_BxM: jax.Array,
      deterministic: bool = True,
      mem_step_BxM: Optional[jax.Array] = None,
  ) -> jax.Array:
    cfg = self.cfg
    Dh = cfg.D // cfg.H

    keys_BxMxD = mem_BxMxD
    if cfg.stm_cross_pos_mode == 'sinusoidal':
      # Without step ids the entries cannot be grouped, so fall back to
      # serial -- which grouping also reduces to at one entry per sentence.
      if (cfg.multi_eos_enabled and cfg.multi_eos_pos_mode == 'grouped'
          and mem_step_BxM is not None):
        pos_BxM = grouped_memory_positions(mem_valid_BxM, mem_step_BxM)
      else:
        pos_BxM = memory_positions(mem_valid_BxM)
      keys_BxMxD = keys_BxMxD + sinusoidal_key_pe(
          pos_BxM,
          cfg.D,
          positional_weight=cfg.stm_positional_weight,
          dtype=mem_BxMxD.dtype,
      )

    multilinear = partial(
        nn.DenseGeneral,
        axis=-1,
        features=(cfg.H, Dh),
        kernel_init=tg_init('attn_in_proj', cfg),
        bias_init=nn.initializers.zeros,
        use_bias=True,
        dtype=cfg.dtype,
    )
    q_BxLxHxDh = multilinear(name='query')(x_BxLxD)
    k_BxMxHxDh = multilinear(name='key')(keys_BxMxD)
    # NOTE: values see the raw memory vectors, without the positional encoding.
    v_BxMxHxDh = multilinear(name='value')(mem_BxMxD)

    # 1/sqrt(Dh) normally; muP's '8_over_d' mode uses 1/Dh so that
    # attention logits stay O(1) as width grows.
    q_BxLxHxDh = q_BxLxHxDh * cfg.attention_logit_scale(Dh)
    att_BxHxLxM = jnp.einsum('...qhd,...khd->...hqk', q_BxLxHxDh, k_BxMxHxDh)
    att_BxHxLxM = att_BxHxLxM.astype(jnp.float32)

    # Mask empty slots. Rows with no memory at all end up with every score at
    # `finfo.min`, which softmaxes to a uniform (finite, NaN-free) distribution;
    # their output is zeroed below, exactly as the reference does.
    neg_inf = jnp.finfo(att_BxHxLxM.dtype).min
    mask_Bx1x1xM = mem_valid_BxM[:, None, None, :]
    att_BxHxLxM = jnp.where(mask_Bx1x1xM, att_BxHxLxM, neg_inf)
    att_BxHxLxM = jax.nn.softmax(att_BxHxLxM, axis=-1)
    att_BxHxLxM = att_BxHxLxM.astype(cfg.dtype)
    att_BxHxLxM = nn.Dropout(rate=cfg.attn_dropout)(
        att_BxHxLxM, deterministic=deterministic
    )

    out_BxLxHxDh = jnp.einsum('...hqk,...khd->...qhd', att_BxHxLxM, v_BxMxHxDh)
    out_BxLxD = nn.DenseGeneral(
        features=cfg.D,
        name='attn_out_proj',
        axis=(-2, -1),
        kernel_init=tg_init('attn_out_proj', cfg),
        bias_init=nn.initializers.zeros,
        use_bias=True,
        dtype=cfg.dtype,
    )(out_BxLxHxDh)

    row_has_mem_B = jnp.any(mem_valid_BxM, axis=-1)
    return out_BxLxD * row_has_mem_B[:, None, None].astype(out_BxLxD.dtype)
