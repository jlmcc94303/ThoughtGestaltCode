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
"""Thought Gestalt (TG): a recurrent sentence-level transformer.

Port of `models/tg/model.py:SentenceTransformer` and
`models/tg/blocks.py:ConfigurableBlock` from the PyTorch reference.

One forward pass handles ONE sentence (`[B, 66]`). Sentences of a document are
processed in order by `run_sentence_loop`, which threads a short-term memory
(STM) of previous sentence vectors (S_REPs) through the steps. Tokens of
sentence `t` cross-attend to the S_REPs of sentences `< t`, so the loss of a
later sentence backpropagates into earlier ones.

CRITICAL: there is no `jax.lax.stop_gradient` on the S_REP -> STM path unless
`cfg.detach_sreps_for_memory` is explicitly set (an ablation, default False).
"""

# pylint: disable=invalid-name,g-importing-member

from functools import partial
from typing import Any, Callable, Optional

from flax import linen as nn
from flax.struct import dataclass
import jax
import jax.numpy as jnp
from tg.models.tg_config import BLOCK_TYPES
from tg.models.tg_config import TgConfig
from tg.models.tg_config import tg_init
from tg.models.tg_cross_attention import TgCrossAttn
from tg.models.tg_srep_head import SrepHead

PyTree = Any


@dataclass
class TgMemory:
  """Short-term memory: up to M S_REPs, ordered oldest -> newest in a prefix."""

  kv_BxMxD: jax.Array
  valid_BxM: jax.Array
  # Sentence step that produced each entry. Only read by
  # `multi_eos_pos_mode="grouped"`, where the K vectors of one sentence must
  # share a cross-attention position; -1 in empty slots.
  step_BxM: jax.Array


def content_token_mask(
    ids_BxL: jax.Array, mask_BxL: jax.Array, cfg: TgConfig
) -> jax.Array:
  """True on content tokens: not padding, not [BOS], not any [EOS], not [EOD].

  Port of `model.py:_content_token_mask`, which excludes `head_token_ids`
  ([BOS]) and `tail_token_ids` (every EOS id plus [EOD]).
  """
  keep = (mask_BxL == 1) & (ids_BxL != cfg.bos_id) & (ids_BxL != cfg.eod_id)
  for tok in tuple(cfg.multi_eos_ids or (cfg.eos_id,)):
    keep = keep & (ids_BxL != int(tok))
  return keep


@dataclass
class TgStepOutput:
  """What one sentence step produces."""

  logits_BxLxV: jax.Array
  srep_BxD: jax.Array
  srep_raw_norm_B: jax.Array
  srep_norm_penalty_B: jax.Array
  has_eos_B: jax.Array
  # Multi-EOS only: all K sentence vectors, [B, K, D], and which of them exist,
  # [B, K]. Under the default (K = 1) these are `srep_BxD[:, None]` and
  # `has_eos_B[:, None]`, so the single-EOS path is unchanged.
  sreps_BxKxD: Optional[jax.Array] = None
  has_eos_BxK: Optional[jax.Array] = None


# --------------------------------------------------------------------------- #
# STM helpers (pure functions; fixed shapes so the per-step fn stays jittable)
# --------------------------------------------------------------------------- #
def init_memory(B: int, cfg: TgConfig) -> TgMemory:
  """Empty STM. The reference resets the deque per document/stream."""
  return TgMemory(
      kv_BxMxD=jnp.zeros((B, cfg.M, cfg.D), dtype=cfg.dtype),
      valid_BxM=jnp.zeros((B, cfg.M), dtype=jnp.bool_),
      step_BxM=jnp.full((B, cfg.M), -1, dtype=jnp.int32),
  )


def push_memory(
    mem: TgMemory,
    srep_BxD: jax.Array,
    write_B: jax.Array,
    step: int = 0,
    backprop_window: Optional[int] = None,
) -> TgMemory:
  """Append one S_REP per row, evicting the oldest entry when full.

  `_append_stm_entry` appends to a `deque` capped at
  `max_sentences_in_short_term`, popping the oldest. We keep occupied slots in
  a prefix ordered oldest -> newest, so appending writes slot `K` (or rolls the
  buffer left and writes the last slot when full). Rows with `write_B == False`
  are left untouched.
  """
  kv_BxMxD, valid_BxM = mem.kv_BxMxD, mem.valid_BxM
  M = kv_BxMxD.shape[1]
  srep_BxD = srep_BxD.astype(kv_BxMxD.dtype)

  k_B = jnp.sum(valid_BxM.astype(jnp.int32), axis=-1)
  full_B = k_B >= M

  kv_rolled_BxMxD = jnp.concatenate(
      [kv_BxMxD[:, 1:], jnp.zeros_like(kv_BxMxD[:, :1])], axis=1
  )
  valid_rolled_BxM = jnp.concatenate(
      [valid_BxM[:, 1:], jnp.zeros_like(valid_BxM[:, :1])], axis=1
  )
  step_BxM = mem.step_BxM
  step_rolled_BxM = jnp.concatenate(
      [step_BxM[:, 1:], jnp.full_like(step_BxM[:, :1], -1)], axis=1
  )
  kv_base_BxMxD = jnp.where(full_B[:, None, None], kv_rolled_BxMxD, kv_BxMxD)
  valid_base_BxM = jnp.where(full_B[:, None], valid_rolled_BxM, valid_BxM)
  step_base_BxM = jnp.where(full_B[:, None], step_rolled_BxM, step_BxM)

  slot_B = jnp.where(full_B, M - 1, k_B)
  onehot_BxM = jax.nn.one_hot(slot_B, M, dtype=kv_BxMxD.dtype)
  onehot_BxMx1 = onehot_BxM[..., None]
  kv_new_BxMxD = (
      kv_base_BxMxD * (1.0 - onehot_BxMx1) + srep_BxD[:, None, :] * onehot_BxMx1
  )
  valid_new_BxM = valid_base_BxM | (onehot_BxM > 0)
  is_slot_BxM = onehot_BxM > 0
  step_new_BxM = jnp.where(is_slot_BxM, jnp.int32(step), step_base_BxM)

  # Truncated backprop. The reference's deque is newest-first and detaches at
  # index >= window; our prefix is oldest-first, so slot j of k has age
  # k-1-j. Re-applied every push, as the reference re-sanitizes every append.
  if backprop_window is not None:
    k_new_B = jnp.sum(valid_new_BxM.astype(jnp.int32), axis=-1)
    age_BxM = (k_new_B[:, None] - 1) - jnp.arange(M, dtype=jnp.int32)[None, :]
    keep_BxM = valid_new_BxM & (age_BxM < int(backprop_window))
    kv_new_BxMxD = jnp.where(
        keep_BxM[..., None], kv_new_BxMxD,
        jax.lax.stop_gradient(kv_new_BxMxD))

  write_BxM = jnp.broadcast_to(write_B[:, None], valid_BxM.shape)
  return TgMemory(
      kv_BxMxD=jnp.where(write_BxM[..., None], kv_new_BxMxD, kv_BxMxD),
      valid_BxM=jnp.where(write_BxM, valid_new_BxM, valid_BxM),
      step_BxM=jnp.where(write_BxM, step_new_BxM, mem.step_BxM),
  )


# --------------------------------------------------------------------------- #
# Modules
# --------------------------------------------------------------------------- #
class TgMlp(nn.Module):
  """Feed-forward network; present in every block (`models/ffn.py:build_ffn`)."""

  cfg: TgConfig

  @nn.compact
  def __call__(self, x_BxLxD: jax.Array, deterministic: bool = True):
    cfg = self.cfg
    linear = partial(
        nn.Dense,
        kernel_init=tg_init('mlp_kernel', cfg),
        bias_init=nn.initializers.zeros,
        use_bias=True,
        dtype=cfg.dtype,
    )
    drop = partial(nn.Dropout(rate=cfg.dropout), deterministic=deterministic)
    if cfg.ffn_activation.lower() == 'swiglu':
      x_BxLx2F = linear(2 * cfg.F)(x_BxLxD)
      a_BxLxF, b_BxLxF = jnp.split(x_BxLx2F, 2, axis=-1)
      x_BxLxF = jax.nn.silu(a_BxLxF) * b_BxLxF
    else:
      x_BxLxF = linear(cfg.F)(x_BxLxD)
      # `nn.GELU()` in torch is the exact (erf) gelu, not the tanh approximation.
      x_BxLxF = jax.nn.gelu(x_BxLxF, approximate=False)
    x_BxLxF = drop(x_BxLxF)
    x_BxLxD = linear(cfg.D)(x_BxLxF)
    return drop(x_BxLxD)


class TgSelfAttn(nn.Module):
  """Causal self-attention over the tokens of a single sentence."""

  cfg: TgConfig

  @nn.compact
  def __call__(
      self,
      x_BxLxD: jax.Array,
      key_pad_BxL: jax.Array,
      deterministic: bool = True,
  ) -> jax.Array:
    cfg = self.cfg
    Dh = cfg.D // cfg.H

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
    k_BxLxHxDh = multilinear(name='key')(x_BxLxD)
    v_BxLxHxDh = multilinear(name='value')(x_BxLxD)

    # 1/sqrt(Dh) normally; muP's '8_over_d' mode uses 1/Dh so that
    # attention logits stay O(1) as width grows.
    q_BxLxHxDh = q_BxLxHxDh * cfg.attention_logit_scale(Dh)
    att_BxHxLxL = jnp.einsum('...qhd,...khd->...hqk', q_BxLxHxDh, k_BxLxHxDh)
    att_BxHxLxL = att_BxHxLxL.astype(jnp.float32)

    L = x_BxLxD.shape[1]
    causal_1x1xLxL = jnp.tril(jnp.ones((1, 1, L, L), dtype=jnp.bool_))
    # `_build_block_mask`: blocked = causal | key_padding.
    allowed_BxHxLxL = jnp.logical_and(
        causal_1x1xLxL, jnp.logical_not(key_pad_BxL)[:, None, None, :]
    )
    neg_inf = jnp.finfo(att_BxHxLxL.dtype).min
    att_BxHxLxL = jnp.where(allowed_BxHxLxL, att_BxHxLxL, neg_inf)
    att_BxHxLxL = jax.nn.softmax(att_BxHxLxL, axis=-1)
    att_BxHxLxL = att_BxHxLxL.astype(cfg.dtype)
    att_BxHxLxL = nn.Dropout(rate=cfg.attn_dropout)(
        att_BxHxLxL, deterministic=deterministic
    )

    out_BxLxHxDh = jnp.einsum('...hqk,...khd->...qhd', att_BxHxLxL, v_BxLxHxDh)
    return nn.DenseGeneral(
        features=cfg.D,
        name='attn_out_proj',
        axis=(-2, -1),
        kernel_init=tg_init('attn_out_proj', cfg),
        bias_init=nn.initializers.zeros,
        use_bias=True,
        dtype=cfg.dtype,
    )(out_BxLxHxDh)


class TgBlock(nn.Module):
  """Pre-LN block. Port of `models/tg/blocks.py:ConfigurableBlock`.

  Five orderings, all ending in an FFN:

    S   self-attention only        (within-sentence, causal)
    C   cross-attention only       (sentence tokens -> STM)
    SC  self THEN cross, serial    -- `ln_mem` sees the POST-self-attention x
    CS  cross THEN self, serial
    P   parallel: both pre-LNs read the SAME input x, increments summed

  `SC`/`CS`/`P` put both attentions in every layer, so they roughly double the
  attention parameters -- the paper's "increasing per-layer capacity" ablation
  (§3.5), where Self->Cross is the best variant and is described as an
  effective route to scale TG.

  A `C` sub-layer whose STM is empty contributes exactly zero (see
  `TgCrossAttn`), so `C` degenerates to FFN-only and `P` degenerates to `S`,
  matching the reference's `if k_memory_keys is None` early-outs.
  """

  cfg: TgConfig
  block_type: str

  @nn.compact
  def __call__(
      self,
      x_BxLxD: jax.Array,
      key_pad_BxL: jax.Array,
      mem_kv_BxMxD: jax.Array,
      mem_valid_BxM: jax.Array,
      deterministic: bool = True,
      mem_step_BxM: Optional[jax.Array] = None,
  ) -> jax.Array:
    cfg = self.cfg
    ln = partial(nn.LayerNorm, dtype=cfg.dtype, epsilon=cfg.layer_norm_epsilon)
    drop = partial(nn.Dropout(rate=cfg.dropout), deterministic=deterministic)
    bt = self.block_type
    if bt not in BLOCK_TYPES:
      raise ValueError(f'Unsupported block type {bt!r}; expected one of '
                       f'{BLOCK_TYPES}')

    def self_increment(x):
      # No self_gate parameter: args_tg.py sets --freeze_self_gates_to_one=True,
      # so the reference's gate is a constant 1.0 with requires_grad=False.
      q = ln(name='ln_self')(x)
      return drop(TgSelfAttn(cfg, name='self_attn')(q, key_pad_BxL,
                                                    deterministic))

    def cross_increment(x):
      q = ln(name='ln_mem')(x)
      m = TgCrossAttn(cfg, name='cross_attn')(q, mem_kv_BxMxD, mem_valid_BxM,
                                              deterministic, mem_step_BxM)
      memory_gate = self.param(
          'memory_gate', lambda _: jnp.asarray(cfg.memory_gate_init, cfg.dtype)
      )
      return drop(memory_gate * m)

    if bt == 'S':
      x_BxLxD = x_BxLxD + self_increment(x_BxLxD)
    elif bt == 'C':
      x_BxLxD = x_BxLxD + cross_increment(x_BxLxD)
    elif bt == 'SC':
      x_BxLxD = x_BxLxD + self_increment(x_BxLxD)
      x_BxLxD = x_BxLxD + cross_increment(x_BxLxD)   # sees updated x
    elif bt == 'CS':
      x_BxLxD = x_BxLxD + cross_increment(x_BxLxD)
      x_BxLxD = x_BxLxD + self_increment(x_BxLxD)    # sees updated x
    else:  # 'P'
      # Both pre-LNs read the ORIGINAL x; this is what distinguishes P from SC.
      self_inc = self_increment(x_BxLxD)
      cross_inc = cross_increment(x_BxLxD)
      x_BxLxD = x_BxLxD + self_inc + cross_inc

    z_BxLxD = ln(name='ln_ffn')(x_BxLxD)
    return x_BxLxD + TgMlp(cfg, name='mlp')(z_BxLxD, deterministic)


class ThoughtGestaltDo(nn.Module):
  """TG decoder over one sentence, conditioned on the STM."""

  cfg: TgConfig

  def setup(self):
    cfg = self.cfg
    self.embed = nn.Embed(
        num_embeddings=cfg.V,
        features=cfg.D,
        embedding_init=tg_init('embedding', cfg),
    )
    # Covers memory prefix + sentence when in-context (lengths.py:
    # max_position_embeddings = sentence_span + effective_slots).
    self.pos_embed = nn.Embed(
        num_embeddings=cfg.L_full,
        features=cfg.D,
        embedding_init=tg_init('embedding', cfg),
    )
    block = nn.remat(TgBlock, static_argnums=(5,)) if cfg.remat else TgBlock
    self.blocks = [
        block(cfg, block_type=cfg.block_config[i]) for i in range(cfg.N)
    ]
    self.out_ln = nn.LayerNorm(dtype=cfg.dtype, epsilon=cfg.layer_norm_epsilon)
    self.srep_head = SrepHead(cfg)
    self.embed_drop = nn.Dropout(rate=cfg.dropout)

  def __call__(
      self,
      ids_BxL: jax.Array,
      mask_BxL: jax.Array,
      mem_kv_BxMxD: jax.Array,
      mem_valid_BxM: jax.Array,
      bos_ctx_BxD: jax.Array,
      bos_ctx_valid_B: jax.Array,
      deterministic: bool = True,
      mem_step_BxM: Optional[jax.Array] = None,
  ) -> TgStepOutput:
    cfg = self.cfg
    tok_BxLxD = self.embed(ids_BxL)

    # BOS seeding ("copy" mode): the previous sentence's S_REP replaces this
    # sentence's BOS embedding, before positional embeddings are added.
    if cfg.bos_replacement_mode == 'copy':
      bos_slot_BxD = jnp.where(
          bos_ctx_valid_B[:, None],
          bos_ctx_BxD.astype(tok_BxLxD.dtype),
          tok_BxLxD[:, 0, :],
      )
      tok_BxLxD = tok_BxLxD.at[:, 0, :].set(bos_slot_BxD)

    L_s = ids_BxL.shape[1]
    span = cfg.memory_span
    # Sentence positions are OFFSET by the memory span in in-context mode
    # (model.py L851: pos_ids_full[:, span : span + ids.size(1)]).
    pos_1xLxD = self.pos_embed(jnp.arange(span, span + L_s)[None, :])
    h_BxLxD = tok_BxLxD + pos_1xLxD

    # Static `token_dropout` is tested first so that `deterministic` is never
    # bool-converted when token dropout is off -- it may legitimately arrive as
    # a tracer (e.g. under a jit that traces every argument). This mirrors
    # `nn.Dropout`, which short-circuits on `rate == 0.0` before touching
    # `deterministic`.
    if cfg.token_dropout_now > 0.0 and not deterministic:
      h_BxLxD = self._token_dropout(h_BxLxD, ids_BxL, mask_BxL)
    h_BxLxD = self.embed_drop(h_BxLxD, deterministic=deterministic)

    key_pad_BxL = mask_BxL == 0

    if span > 0:
      # In-context memory: PREPEND the memory vectors as tokens and let causal
      # self-attention read them. `_build_memory_prefix` orders them
      # oldest -> newest, which is how TgMemory already stores them, and gives
      # empty slots a padding mask.
      mem_prefix = mem_kv_BxMxD[:, :span].astype(h_BxLxD.dtype)
      mem_pos = self.pos_embed(jnp.arange(0, span)[None, :])
      h_BxLxD = jnp.concatenate([mem_prefix + mem_pos, h_BxLxD], axis=1)
      mem_pad = jnp.logical_not(mem_valid_BxM[:, :span])
      key_pad_BxL = jnp.concatenate([mem_pad, key_pad_BxL], axis=1)

    h_srep_BxLxD = None
    for i, block in enumerate(self.blocks):
      h_BxLxD = block(
          h_BxLxD, key_pad_BxL, mem_kv_BxMxD, mem_valid_BxM, deterministic,
          mem_step_BxM
      )
      if i == cfg.srep_layer_idx:
        h_srep_BxLxD = h_BxLxD

    h_BxLxD = self.out_ln(h_BxLxD)
    if h_srep_BxLxD is None:  # srep_extraction_layer < 0 => final layer
      h_srep_BxLxD = h_BxLxD
    if span > 0:
      # Drop the memory prefix so logits/S_REP are indexed by SENTENCE position.
      # The reference keeps the full sequence and masks memory positions out of
      # the loss (`loss_mask_full`); slicing is equivalent, because the only
      # transition it removes is memory[-1] -> BOS, which that mask excludes.
      h_BxLxD = h_BxLxD[:, span:]
      h_srep_BxLxD = h_srep_BxLxD[:, span:]
    logits_BxLxV = self.embed.attend(h_BxLxD.astype(jnp.float32))

    # S_REP is read at the first [EOS]; `_first_positions_for_ids` falls back to
    # position 0 when the sentence has none (argmax already returns 0 there).
    # Under multi-EOS the same is done for each of [EOS], [EOS2], ... in turn,
    # producing K vectors per sentence (`model.py` L1146: `for eos_idx,
    # token_idx_sentence in enumerate(eos_positions)`).
    eos_ids = tuple(cfg.multi_eos_ids or (cfg.eos_id,))

    # mean_pool: average CONTENT tokens instead of reading [EOS], falling back
    # to [EOS] for an empty sentence. Under multi-EOS every EOS gets the same
    # mean, since the branch does not depend on which EOS is processed.
    mean_pool_BxD = None
    has_content_B = None
    if cfg.srep_pool_mode == 'mean_pool':
      # The memory prefix was sliced off above, so h_srep is sentence-indexed
      # and aligns with ids_BxL; no memory_span offset needed.
      content_BxL = content_token_mask(ids_BxL, mask_BxL, cfg)
      w_BxL = content_BxL.astype(h_srep_BxLxD.dtype)
      n_B = jnp.sum(w_BxL, axis=-1)
      has_content_B = n_B > 0
      mean_pool_BxD = (
          jnp.einsum('bl,bld->bd', w_BxL, h_srep_BxLxD)
          / jnp.maximum(n_B, 1.0)[:, None]
      )

    sreps, raw_norms, penalties, founds = [], [], [], []
    for tok in eos_ids:
      hit_BxL = ids_BxL == int(tok)
      found_B = jnp.any(hit_BxL, axis=-1)
      pos_B = jnp.argmax(hit_BxL.astype(jnp.int32), axis=-1)
      h_tok_BxD = jnp.take_along_axis(
          h_srep_BxLxD, pos_B[:, None, None], axis=1
      )[:, 0, :]
      if mean_pool_BxD is not None:
        h_tok_BxD = jnp.where(has_content_B[:, None], mean_pool_BxD, h_tok_BxD)
      out = self.srep_head(h_tok_BxD, deterministic=deterministic)
      sreps.append(out.srep_BxD)
      raw_norms.append(out.raw_norm_B)
      penalties.append(out.norm_penalty_B)
      founds.append(found_B)

    sreps_BxKxD = jnp.stack(sreps, axis=1)
    has_eos_BxK = jnp.stack(founds, axis=1)
    # The primary ([EOS]) vector drives BOS seeding and the single-EOS path;
    # the reference gates seeding on `is_primary_eos = (eos_idx == 0)`.
    return TgStepOutput(
        logits_BxLxV=logits_BxLxV,
        srep_BxD=sreps[0],
        srep_raw_norm_B=jnp.mean(jnp.stack(raw_norms, axis=1), axis=1),
        srep_norm_penalty_B=jnp.mean(jnp.stack(penalties, axis=1), axis=1),
        has_eos_B=founds[0],
        sreps_BxKxD=sreps_BxKxD,
        has_eos_BxK=has_eos_BxK,
    )

  def _token_dropout(
      self, h_BxLxD: jax.Array, ids_BxL: jax.Array, mask_BxL: jax.Array
  ) -> jax.Array:
    """Zero whole content-token embeddings (no rescaling), as in the reference."""
    cfg = self.cfg
    content_BxL = content_token_mask(ids_BxL, mask_BxL, cfg)
    rand_BxL = jax.random.uniform(self.make_rng('dropout'), ids_BxL.shape)
    drop_BxL = (rand_BxL < cfg.token_dropout_now) & content_BxL
    return jnp.where(drop_BxL[..., None], 0.0, h_BxLxD)


# --------------------------------------------------------------------------- #
# Per-document sentence loop
# --------------------------------------------------------------------------- #
def run_sentence_loop(
    apply_fn: Callable[..., TgStepOutput],
    variables: PyTree,
    sentences_BxTxL: jax.Array,
    masks_BxTxL: jax.Array,
    lengths_B: jax.Array,
    cfg: TgConfig,
    *,
    step_fn: Callable[..., PyTree],
    deterministic: bool = True,
    dropout_rng: Optional[jax.Array] = None,
) -> PyTree:
  """Run every sentence of a batch of documents, threading the STM.

  Mirrors `SentenceTransformer.sentence_steps`: a plain Python loop over
  sentence steps (`T` is static), where each step is a fixed-shape,
  jit-friendly call to `apply_fn`. `step_fn(t, out, ids, mask, row_valid)`
  returns a pytree that is summed over steps, mirroring the reference's
  `forward(documents, step_fn=...)` so per-step logits stay transient.

  Rows of the batch are documents; `lengths_B` gives each one's sentence count
  so shorter documents are masked out of later steps.
  """
  B, T, _ = sentences_BxTxL.shape
  # Truncated backprop through memory; None keeps the full graph (default).
  backprop_window = cfg.stm_backprop_window_effective
  mem = init_memory(B, cfg)
  bos_ctx_BxD = jnp.zeros((B, cfg.D), dtype=cfg.dtype)
  bos_ctx_valid_B = jnp.zeros((B,), dtype=jnp.bool_)
  acc = None

  for t in range(T):
    ids_BxL = sentences_BxTxL[:, t]
    mask_BxL = masks_BxTxL[:, t]
    row_valid_B = t < lengths_B

    rngs = None
    step_rng = None
    if not deterministic:
      if dropout_rng is None:
        raise ValueError('dropout_rng is required when deterministic=False')
      step_rng = jax.random.fold_in(dropout_rng, t)
      rngs = {'dropout': step_rng}

    out = apply_fn(
        variables,
        ids_BxL,
        mask_BxL,
        mem.kv_BxMxD,
        mem.valid_BxM,
        bos_ctx_BxD,
        bos_ctx_valid_B,
        deterministic,
        mem.step_BxM,
        rngs=rngs,
    )

    contrib = step_fn(t, out, ids_BxL, mask_BxL, row_valid_B)
    acc = contrib if acc is None else jax.tree_util.tree_map(jnp.add, acc,
                                                             contrib)

    # Write this sentence's S_REP into the STM. No stop_gradient here: sentence
    # t+1's loss must reach sentence t through this tensor.
    srep_BxD = out.srep_BxD
    # `memory_dropout`: zero an ENTIRE memory entry with probability p, no
    # rescaling (models/tg/model.py L1180-1183). Applied to the vector stored
    # in the STM; the copy used for BOS seeding is unaffected, matching the
    # reference. Reference default is 0.0, i.e. numerically inert.
    if (not deterministic) and cfg.memory_dropout_now > 0.0:
      keep_B = (
          jax.random.uniform(
              jax.random.fold_in(step_rng, 991), (srep_BxD.shape[0],)
          )
          >= cfg.memory_dropout_now
      )
      srep_BxD_mem_src = srep_BxD * keep_B[:, None].astype(srep_BxD.dtype)
    else:
      srep_BxD_mem_src = srep_BxD
    if cfg.detach_sreps_for_memory:
      srep_mem_BxD = jax.lax.stop_gradient(srep_BxD_mem_src)
    else:
      srep_mem_BxD = srep_BxD_mem_src
    # `if token_is_eos:` guards the append in the reference.
    write_B = row_valid_B & out.has_eos_B
    if cfg.use_memory:
      if cfg.multi_eos_enabled and out.sreps_BxKxD is not None:
        # One entry per EOS, oldest-first in the order the tokens appear, so
        # the K vectors of a sentence occupy K consecutive STM slots.
        K = out.sreps_BxKxD.shape[1]
        for k in range(K):
          vec_BxD = out.sreps_BxKxD[:, k]
          if (not deterministic) and cfg.memory_dropout_now > 0.0:
            keep_B = (
                jax.random.uniform(
                    jax.random.fold_in(step_rng, 991 + k),
                    (vec_BxD.shape[0],))
                >= cfg.memory_dropout_now
            )
            vec_BxD = vec_BxD * keep_B[:, None].astype(vec_BxD.dtype)
          if cfg.detach_sreps_for_memory:
            vec_BxD = jax.lax.stop_gradient(vec_BxD)
          mem = push_memory(
              mem, vec_BxD, row_valid_B & out.has_eos_BxK[:, k],
              step=t, backprop_window=backprop_window)
      else:
        mem = push_memory(mem, srep_mem_BxD, write_B, step=t,
                          backprop_window=backprop_window)

    if cfg.bos_replacement_mode == 'copy':
      if cfg.bos_context_detach or cfg.detach_sreps_for_memory:
        bos_ctx_BxD = jax.lax.stop_gradient(srep_BxD)
      else:
        bos_ctx_BxD = srep_BxD
      # Only seeded when a next sentence exists in the same document.
      bos_ctx_valid_B = write_B & ((t + 1) < lengths_B)

  return acc
