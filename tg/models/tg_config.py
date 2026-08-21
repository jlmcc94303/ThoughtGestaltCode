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
"""Config for the Thought Gestalt (TG) sentence-level recurrent model.

Ported from `models/tg/config.py` in the PyTorch reference, with the *actual*
training defaults taken from `train_pipelines/tg/args_tg.py` (those override the
dataclass defaults in the reference config).
"""

# pylint: disable=invalid-name

import dataclasses
import math
from typing import Any, Optional, Sequence

from flax import linen as nn
import jax.numpy as jnp


def _round_to_multiple(value: float, multiple: int) -> int:
  """Mirror of `models/ffn.py:_round_to_multiple`."""
  if multiple <= 0:
    return int(round(value))
  rounded = int(round(value / multiple) * multiple)
  return max(multiple, rounded)


def compute_ffn_dim(D: int, activation: str, align_to: int = 8) -> int:
  """Mirror of `models/ffn.py:compute_ffn_dim`."""
  base = int(4 * D)
  act = (activation or 'gelu').lower()
  if act == 'swiglu':
    return _round_to_multiple((2.0 / 3.0) * float(base), align_to)
  if act == 'gelu':
    return base
  raise ValueError(f'Unsupported FFN activation: {activation}')


def compute_head_dim(base_dim: int, activation: str, align_to: int = 8) -> int:
  """Mirror of `models/ffn.py:compute_head_dim`."""
  act = (activation or 'gelu').lower()
  if act == 'swiglu':
    return _round_to_multiple((2.0 / 3.0) * float(base_dim), align_to)
  if act == 'gelu':
    return int(base_dim)
  raise ValueError(f'Unsupported FFN activation: {activation}')


# GPT-2 (50257) + the four specials added by `core/tokenizer_setup.py`, in the
# order HF assigns them: [PAD], [BOS], [EOS], [EOD].
GPT2_VOCAB_SIZE = 50257
DEFAULT_PAD_ID = 50257
DEFAULT_BOS_ID = 50258
DEFAULT_EOS_ID = 50259
DEFAULT_EOD_ID = 50260
DEFAULT_VOCAB_SIZE = 50261

# args_tg.py --block_config default.
DEFAULT_BLOCK_CONFIG = ('S', 'C') * 6

# models/tg/blocks.py:ConfigurableBlock supports five orderings. 'SC'/'CS'/'P'
# place BOTH attentions in every layer (the paper's per-layer-capacity ablation,
# §3.5); 'S'/'C' place one each and alternate.
BLOCK_TYPES = ('S', 'C', 'SC', 'CS', 'P')


@dataclasses.dataclass
class TgConfig:
  """Hyper-parameters for the TG recurrent sentence transformer."""

  # --- Geometry (args_tg.py: --d_model/--n_heads/--n_layers) ---------------
  D: int = 768  # model/embed dim = qkv dim
  H: int = 12  # num attention heads
  N: int = 12  # number of blocks
  V: int = DEFAULT_VOCAB_SIZE  # vocab size
  F: Optional[int] = None  # FF inner dim; None => derived from ffn_activation

  # --- Sentence layout (transform.py / lengths.py) -------------------------
  # sentence_span = 1 (BOS) + max_sentence_tokens + sentence_tail_len = 66.
  max_sentence_tokens: int = 64
  sentence_tail_len: int = 1

  # --- Segmentation --------------------------------------------------------
  # 'span' compresses fixed token windows instead of sentences, and overrides
  # max_sentence_tokens/sentence_tail_len below.
  segmentation: str = 'sentence'  # 'sentence' | 'span'
  span_tokens: int = 25

  # --- Short-term memory ---------------------------------------------------
  max_sentences_in_short_term: int = 40  # M
  # model_setup.py resolves --stm_cross_pos_mode to "sinusoidal" because
  # --use_stm_positional defaults to True.
  stm_cross_pos_mode: str = 'sinusoidal'  # 'sinusoidal' | 'none'
  stm_positional_weight: float = 1.0
  # 'external' -> cross-attention to memory, O(n*M). 'in_context' -> memory
  # prepended as tokens, O((n+M)^2), which also forces every block to type 'S'.
  memory_mode: str = 'external'  # 'external' | 'in_context'
  context_memory_slots: Optional[int] = None  # None => M when in_context
  # Ablation switches. `use_memory=False` mirrors --debug_no_memory: the STM is
  # never written, so every C block's cross-attention contributes exactly zero.
  use_memory: bool = True
  # Mirrors --ablate_detach_sentence_reps. MUST stay False for the recurrence
  # to train: gradients have to flow from later sentences back through memory.
  detach_sreps_for_memory: bool = False
  # Truncated backprop through the STM: only the newest N entries keep their
  # graph. None -> unlimited (default), 0 -> all detached, N>0 -> newest N.
  stm_backprop_window: Optional[int] = None

  # --- Blocks --------------------------------------------------------------
  block_config: Sequence[str] = DEFAULT_BLOCK_CONFIG
  ffn_activation: str = 'swiglu'  # args_tg.py --ffn_activation
  layer_norm_epsilon: float = 1e-6

  # --- Gates ---------------------------------------------------------------
  # No self-attention gate: the reference freezes it at a constant 1.0, so only
  # memory_gate is learnable.
  memory_gate_init: float = 1.0

  # --- Dropout -------------------------------------------------------------
  # dropout and attn_dropout are active from step 0; token/memory/srep are
  # scaled by the warm-in multiplier (see `dropout_scale`).
  dropout: float = 0.1
  attn_dropout: float = 0.2
  srep_dropout: float = 0.15   # warmed
  token_dropout: float = 0.15  # warmed
  memory_dropout: float = 0.0  # warmed
  # Warm-in multiplier applied to token/memory/srep dropout ONLY. Held static
  # per phase so it is a compile-time constant; the trainer rebuilds the module
  # at the phase boundaries below (train_loop.py L626-655).
  dropout_scale: float = 1.0
  dropout_half_step: int = 2000  # args_tg.py --dropout_half_step
  dropout_full_step: int = 7000  # args_tg.py --dropout_full_step

  # --- S_REP (args_tg.py) --------------------------------------------------
  srep_extraction_layer: int = 6  # 0-indexed block whose output feeds the head
  srep_head_depth: int = 1
  srep_head_mult: int = 1
  srep_norm_target: float = 1.0
  srep_norm_margin: float = 0.1
  srep_norm_reg_weight: float = 0.01
  # 'mean_pool' averages CONTENT tokens (no [BOS]/[EOS]/[EOD]/padding),
  # falling back to the [EOS] state for a sentence with none.
  srep_pool_mode: str = 'eos'  # 'eos' | 'mean_pool'

  # --- BOS seeding ---------------------------------------------------------
  bos_replacement_mode: str = 'copy'  # 'copy' | 'off'
  bos_context_detach: bool = False

  # --- Special token ids (resolve from the tokenizer at runtime) -----------
  bos_id: int = DEFAULT_BOS_ID
  eos_id: int = DEFAULT_EOS_ID
  eod_id: int = DEFAULT_EOD_ID
  pad_id: int = DEFAULT_PAD_ID

  # --- Data ----------------------------------------------------------------
  max_sentences_per_stream: int = 30  # chunk size for long documents

  # --- Init / sharding / precision ----------------------------------------
  # The reference re-inits every Linear/Embedding to normal(0, 0.02); attention
  # in-projections keep xavier_uniform.
  # --- Multi-EOS -----------------------------------------------------------
  # Append K distinct EOS tokens per sentence and write K memory entries
  # instead of one. Off by default and not reported in the paper.
  multi_eos_count: int = 1
  # Positional index given to the K entries of one sentence in cross-attention:
  #   'serial'  -> each entry gets its own index (K distinct positions)
  #   'grouped' -> all K share their sentence's index
  multi_eos_pos_mode: str = 'grouped'
  # 'mask_all' excludes every EOS label from the loss; 'last_only' supervises
  # only the final EOS of each sentence.
  multi_eos_loss_mode: str = 'mask_all'
  # Ids of [EOS], [EOS2], ..., filled in from the tokenizer. None => (eos_id,).
  multi_eos_ids: Optional[Sequence[int]] = None

  # --- muP -----------------------------------------------------------------
  # Scales init std, learning rate and attention logits relative to mup_base_D
  # so hyper-parameters transfer across widths. Hidden matrices only:
  # embeddings, norms, gates and biases keep standard parameterization.
  mup_enabled: bool = False
  mup_base_D: Optional[int] = None  # None => D at construction (no-op scaling)
  mup_attention_scale: str = 'sqrt'  # 'sqrt' | '8_over_d'

  kernel_init: nn.initializers.Initializer = nn.initializers.normal(0.02)
  attn_in_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()
  embed_init: nn.initializers.Initializer = nn.initializers.normal(0.02)
  dtype: Any = jnp.float32
  fsdp_enabled: bool = True
  remat: bool = False

  def __post_init__(self):
    if self.D % self.H != 0:
      raise ValueError(f'D {self.D} not divisible by H {self.H}')
    self.block_config = tuple(str(b).upper() for b in self.block_config)
    if len(self.block_config) != self.N:
      raise ValueError(
          f'block_config length ({len(self.block_config)}) must match N '
          f'({self.N})'
      )
    for b in self.block_config:
      if b not in BLOCK_TYPES:
        raise ValueError(
            f'Unsupported block type {b!r}; expected one of {BLOCK_TYPES}'
        )
    if self.multi_eos_count < 1:
      raise ValueError(
          f'multi_eos_count must be >= 1; got {self.multi_eos_count}')
    if self.multi_eos_pos_mode not in ('serial', 'grouped'):
      raise ValueError(
          f'multi_eos_pos_mode must be "serial" or "grouped"; got '
          f'{self.multi_eos_pos_mode!r}')
    if self.multi_eos_loss_mode not in ('mask_all', 'last_only'):
      raise ValueError(
          f'multi_eos_loss_mode must be "mask_all" or "last_only"; got '
          f'{self.multi_eos_loss_mode!r}')
    if self.multi_eos_count > 1:
      # `args_tg.py` L483-487: the tail must hold K EOS tokens plus the EOD
      # slot, and multi-EOS is incompatible with span segmentation.
      if self.segmentation == 'span':
        raise ValueError(
            'multi_eos_count > 1 is incompatible with segmentation="span" '
            '(args_tg.py rejects --multi_eos_tg with --pseudo_sentence_mode)')
      self.sentence_tail_len = int(self.multi_eos_count) + 1
      if self.multi_eos_ids is not None:
        if len(self.multi_eos_ids) != self.multi_eos_count:
          raise ValueError(
              f'multi_eos_ids has {len(self.multi_eos_ids)} entries but '
              f'multi_eos_count={self.multi_eos_count}')
    if self.multi_eos_ids is None:
      self.multi_eos_ids = (self.eos_id,) * 1 if self.multi_eos_count == 1 else None
      if self.multi_eos_ids is None:
        raise ValueError(
            'multi_eos_ids must be supplied when multi_eos_count > 1 '
            '(build them from the tokenizer with tg_data.multi_eos_ids)')
    self.multi_eos_ids = tuple(int(t) for t in self.multi_eos_ids)
    if self.mup_enabled:
      if self.mup_attention_scale not in ('sqrt', '8_over_d'):
        raise ValueError(
            f'mup_attention_scale must be "sqrt" or "8_over_d"; got '
            f'{self.mup_attention_scale!r}')
      if self.mup_base_D is not None and int(self.mup_base_D) <= 0:
        raise ValueError(f'mup_base_D must be > 0; got {self.mup_base_D}')
      # muP scales hidden matrices only, so it needs the base init std as a
      # reference point; rebuild the kernel init at the scaled width.
      self.kernel_init = nn.initializers.normal(0.02 * self.mup_init_scale)
    if self.segmentation not in ('sentence', 'span'):
      raise ValueError(
          f"segmentation must be 'sentence' or 'span'; got "
          f'{self.segmentation!r}')
    if self.segmentation == 'span':
      # Set here, not at the call site, so L, the data layout and the
      # positional embedding table can never disagree.
      if self.span_tokens <= 0:
        raise ValueError(f'span_tokens must be > 0; got {self.span_tokens}')
      self.max_sentence_tokens = int(self.span_tokens)
      self.sentence_tail_len = 1
      if self.bos_replacement_mode != 'off':
        self.bos_replacement_mode = 'off'  # forced off in span mode
    if self.sentence_tail_len < 1:
      raise ValueError('sentence_tail_len must be >= 1')
    if self.max_sentence_tokens <= 0:
      raise ValueError('max_sentence_tokens must be > 0')
    if self.srep_pool_mode not in ('eos', 'mean_pool'):
      raise ValueError(
          f"srep_pool_mode must be 'eos' or 'mean_pool'; got "
          f'{self.srep_pool_mode!r}')
    if self.bos_replacement_mode not in ('off', 'copy'):
      raise ValueError(
          f'Unsupported bos_replacement_mode={self.bos_replacement_mode}'
      )
    if self.memory_mode not in ('external', 'in_context'):
      raise ValueError(
          f"memory_mode must be 'external' or 'in_context'; got "
          f'{self.memory_mode!r}')
    if self.memory_mode == 'in_context':
      # `models/tg/model.py` L146: in-context memory replaces every block with
      # a plain self-attention block.
      self.block_config = ('S',) * self.N
      if self.context_memory_slots is None:
        self.context_memory_slots = self.max_sentences_in_short_term
    else:
      self.context_memory_slots = 0
    if self.stm_cross_pos_mode not in ('none', 'sinusoidal'):
      raise ValueError(
          f'Unsupported stm_cross_pos_mode={self.stm_cross_pos_mode}'
      )
    if self.stm_cross_pos_mode == 'sinusoidal' and self.D % 2 != 0:
      raise ValueError('D must be even for sinusoidal STM positions.')
    if self.srep_head_depth < 0:
      raise ValueError('srep_head_depth must be >= 0')
    if self.F is None:
      self.F = compute_ffn_dim(self.D, self.ffn_activation)

  # --- Derived -------------------------------------------------------------
  @property
  def L(self) -> int:
    """Fixed sentence length (`lengths.py:sentence_span`); 66 by default."""
    return 1 + self.max_sentence_tokens + self.sentence_tail_len

  @property
  def M(self) -> int:
    """Number of STM slots.

    Under multi-EOS each sentence writes `multi_eos_count` entries, so capacity
    is scaled to keep the memory spanning the same number of *sentences*
    (`model.py:_stm_capacity_entries`).
    """
    return self.max_sentences_in_short_term * int(self.multi_eos_count)

  @property
  def memory_span(self) -> int:
    """Tokens the memory prefix occupies (0 unless in-context)."""
    return int(self.context_memory_slots or 0)

  @property
  def L_full(self) -> int:
    """Sequence length the blocks actually see: memory prefix + sentence.

    `lengths.py:resolve_tg_lengths` -> `max_position_embeddings =
    sentence_span + effective_slots`, so the positional table must cover both.
    """
    return self.memory_span + self.L

  @property
  def srep_layer_idx(self) -> int:
    """Block index whose output feeds the S_REP head (N => use `out_ln`)."""
    if self.srep_extraction_layer < 0:
      return self.N
    return min(self.srep_extraction_layer, self.N - 1)

  @property
  def srep_head_hidden(self) -> int:
    base = max(self.D, int(self.srep_head_mult) * self.D)
    return compute_head_dim(base, self.ffn_activation)

  # --- Warm-in-scaled rates (token / memory / srep only) -------------------
  @property
  def token_dropout_now(self) -> float:
    return self.token_dropout * self.dropout_scale

  @property
  def memory_dropout_now(self) -> float:
    return self.memory_dropout * self.dropout_scale

  @property
  def srep_dropout_now(self) -> float:
    return self.srep_dropout * self.dropout_scale


  @property
  def mup_base_width(self) -> int:
    """Width the muP hyper-parameters were tuned at. Defaults to `D` (no-op)."""
    return int(self.mup_base_D) if self.mup_base_D else int(self.D)

  @property
  def mup_width_ratio(self) -> float:
    """`D / base`. 1.0 when muP is off or the model is at the base width."""
    if not self.mup_enabled:
      return 1.0
    return float(self.D) / float(self.mup_base_width)

  @property
  def mup_init_scale(self) -> float:
    """Multiplier on hidden-matrix init std: `1/sqrt(width_ratio)`."""
    if not self.mup_enabled:
      return 1.0
    return 1.0 / math.sqrt(self.mup_width_ratio)

  @property
  def mup_lr_scale(self) -> float:
    """Multiplier on hidden-matrix learning rate: `1/width_ratio`."""
    if not self.mup_enabled:
      return 1.0
    return 1.0 / self.mup_width_ratio

  @property
  def multi_eos_enabled(self) -> bool:
    return int(self.multi_eos_count) > 1

  @property
  def memory_entries_per_sentence(self) -> int:
    """How many STM slots one sentence consumes."""
    return int(self.multi_eos_count)

  @property
  def stm_backprop_window_effective(self) -> Optional[int]:
    """Port of `model.py:_stm_backprop_window_effective`.

    None -> unlimited · 0 -> all entries detached · N>0 -> newest N keep grads.
    `detach_sreps_for_memory` wins, and the window is clamped to the STM
    capacity (which already includes the multi-EOS multiplier via `M`).
    """
    if self.detach_sreps_for_memory:
      return 0
    if self.stm_backprop_window is None:
      return None
    window = int(self.stm_backprop_window)
    if window < 0:
      return None
    return min(window, int(self.M))

  def attention_logit_scale(self, head_dim: int) -> float:
    """Query scaling. Standard is `1/sqrt(Dh)`; muP can use `1/Dh`."""
    if self.mup_enabled and self.mup_attention_scale == '8_over_d':
      return 1.0 / float(head_dim)
    return 1.0 / math.sqrt(float(head_dim))

def dropout_multiplier_for_step(step: int, half_step: int, full_step: int
                                ) -> float:
  """`train_loop.py` L626-633: 0.0 -> 0.5 -> 1.0 at the two step boundaries."""
  if step < half_step:
    return 0.0
  if step < full_step:
    return 0.5
  return 1.0


def tg_init(layer_type: str, cfg: TgConfig) -> nn.initializers.Initializer:
  """FSDP partitioning + per-layer initializer, mirroring `tg/models/fsdp.py`.

  Unlike `fsdp.init` this lets attention in-projections keep their own
  initializer, matching the PyTorch reference (xavier for `in_proj_weight`,
  normal(0, 0.02) for every `nn.Linear`/`nn.Embedding`).
  """
  partition_fn = nn.with_partitioning if cfg.fsdp_enabled else lambda x, _: x
  if layer_type == 'embedding':  # [V, D] or [L, D]
    return partition_fn(cfg.embed_init, (None, 'data'))
  elif layer_type == 'attn_in_proj':  # [D, H, Dh]
    return partition_fn(cfg.attn_in_init, ('data', None, None))
  elif layer_type == 'attn_out_proj':  # [H, Dh, D]
    return partition_fn(cfg.kernel_init, (None, None, 'data'))
  elif layer_type == 'mlp_kernel':  # [D, F]
    return partition_fn(cfg.kernel_init, ('data', None))
  elif layer_type == 'head':  # [D, D]
    return partition_fn(cfg.kernel_init, ('data', None))
  else:
    raise ValueError(f'unrecognized layer type: {layer_type}')
