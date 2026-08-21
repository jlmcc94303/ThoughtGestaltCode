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
"""S_REP head: the sentence vector written into short-term memory.

Port of `models/tg/srep_head.py:_make_srep_head` plus the S_REP block of
`models/tg/model.py:sentence_steps` (the `ln_srep` / `srep_dropout` /
normalize / hinge-penalty sequence).

The reference applies `ln_srep` to the whole `[B, L, D]` hidden state and then
gathers the EOS position; LayerNorm is per-position so gathering first and
normalizing the `[B, D]` slice is identical.
"""

# pylint: disable=invalid-name,g-importing-member

from flax import linen as nn
from flax.struct import dataclass
import jax
import jax.numpy as jnp
from tg.models.tg_config import TgConfig
from tg.models.tg_config import tg_init


@dataclass
class SrepOutput:
  srep_BxD: jax.Array  # L2-normalized to `srep_norm_target`
  raw_norm_B: jax.Array  # pre-normalization norm (what the hinge sees)
  norm_penalty_B: jax.Array  # unweighted hinge penalty, per row


def srep_norm_penalty(
    raw_norm_B: jax.Array, target: float, margin: float
) -> jax.Array:
  """Squared hinge penalty when `||s||` leaves `[target-margin, target+margin]`.

  Mirrors the penalty branch in `sentence_steps`; the reference weights it by
  `srep_norm_reg_weight` when aggregating (see `tg_loss.py`).
  """
  lower = target - margin
  upper = target + margin
  below_B = jnp.maximum(lower - raw_norm_B, 0.0)
  above_B = jnp.maximum(raw_norm_B - upper, 0.0)
  return below_B**2 + above_B**2


class SrepHead(nn.Module):
  """LayerNorm -> dropout -> MLP -> L2 normalize, with the norm hinge."""

  cfg: TgConfig

  @nn.compact
  def __call__(
      self, h_BxD: jax.Array, deterministic: bool = True
  ) -> SrepOutput:
    cfg = self.cfg
    x_BxD = nn.LayerNorm(
        dtype=cfg.dtype, epsilon=cfg.layer_norm_epsilon, name='ln_srep'
    )(h_BxD)
    x_BxD = nn.Dropout(rate=cfg.srep_dropout_now)(
        x_BxD, deterministic=deterministic
    )
    raw_BxD = self._head(x_BxD, deterministic)

    # `torch.norm(v, p=2)` / `F.normalize(v, p=2, dim=0)` on the per-row vector.
    raw_norm_B = jnp.linalg.norm(raw_BxD.astype(jnp.float32), axis=-1)
    penalty_B = srep_norm_penalty(
        raw_norm_B, cfg.srep_norm_target, cfg.srep_norm_margin
    )
    denom_Bx1 = jnp.maximum(raw_norm_B, 1e-12)[:, None].astype(raw_BxD.dtype)
    srep_BxD = raw_BxD / denom_Bx1 * cfg.srep_norm_target
    return SrepOutput(
        srep_BxD=srep_BxD, raw_norm_B=raw_norm_B, norm_penalty_B=penalty_B
    )

  def _head(self, x_BxD: jax.Array, deterministic: bool) -> jax.Array:
    """SimCLR-style MLP; depth 0 => identity, depth 1 => single Linear."""
    cfg = self.cfg
    depth = int(cfg.srep_head_depth)
    if depth <= 0:
      return x_BxD

    dense = lambda features, name: nn.Dense(
        features=features,
        kernel_init=tg_init('head', cfg),
        bias_init=nn.initializers.zeros,
        use_bias=True,
        dtype=cfg.dtype,
        name=name,
    )
    if depth == 1:  # args_tg.py default (--srep_head_depth 1)
      return dense(cfg.D, 'proj')(x_BxD)

    hidden = cfg.srep_head_hidden
    swiglu = cfg.ffn_activation.lower() == 'swiglu'
    y = x_BxD
    for i in range(depth - 1):
      if swiglu:
        y = dense(2 * hidden, f'block_{i}')(y)
        a, b = jnp.split(y, 2, axis=-1)
        y = jax.nn.silu(a) * b
      else:
        y = dense(hidden, f'block_{i}')(y)
        y = jax.nn.gelu(y, approximate=False)
      y = nn.Dropout(rate=cfg.dropout)(y, deterministic=deterministic)
    return dense(cfg.D, 'proj')(y)
