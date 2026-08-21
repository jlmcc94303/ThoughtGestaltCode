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
"""GPT-2 with gist attention masking (paper 3.3, Table 5).

Structurally identical to `tg/models/gpt2.py` -- same blocks, MLP, init and
parameter names -- except the attention takes an additive `[B, 1, L, L]` bias
restricting each token to its own sentence plus earlier sentences' `[EOS]`
gists.

A separate module rather than a flag so the baseline and its golden-value test
stay byte-for-byte unchanged.
"""

# pylint: disable=invalid-name,g-importing-member

from functools import partial
from typing import Optional

from flax import linen as nn
import jax
import jax.numpy as jnp

from tg.models import fsdp
from tg.models.gpt2 import DoConfig
from tg.models.gpt2 import Mlp


class GistAttn(nn.Module):
  """Causal attention with an additional additive bias."""

  cfg: DoConfig

  @nn.compact
  def __call__(
      self,
      x_BxLxD: jax.Array,
      bias_Bx1xLxL: Optional[jax.Array] = None,
  ):
    cfg = self.cfg
    assert cfg.D % cfg.H == 0, f'D {cfg.D} not divisible by H {cfg.H}'
    Dh = cfg.D // cfg.H

    multilinear = partial(
        nn.DenseGeneral,
        axis=-1,
        features=(cfg.H, Dh),
        kernel_init=fsdp.init('attn_in_proj', cfg),
        use_bias=False,
        dtype=cfg.dtype,
    )

    q_BxLxHxDh, k_BxLxHxDh, v_BxLxHxDh = (
        multilinear(name='query')(x_BxLxD),
        multilinear(name='key')(x_BxLxD),
        multilinear(name='value')(x_BxLxD),
    )
    q_BxLxHxDh /= Dh**0.5
    att_BxHxLxL = jnp.einsum('...qhd,...khd->...hqk', q_BxLxHxDh, k_BxLxHxDh)
    att_BxHxLxL = att_BxHxLxL.astype(jnp.float32)

    L = x_BxLxD.shape[1]
    mask_1x1xLxL = jnp.tril(jnp.ones((1, 1, L, L), dtype=jnp.bool_))
    _NEG_INF = jnp.finfo(cfg.dtype).min
    att_BxHxLxL = jnp.where(mask_1x1xLxL, att_BxHxLxL, _NEG_INF)
    if bias_Bx1xLxL is not None:
      # Additive, so it composes with the causal mask rather than replacing it.
      att_BxHxLxL = att_BxHxLxL + bias_Bx1xLxL.astype(jnp.float32)

    att_BxHxLxL = jax.nn.softmax(att_BxHxLxL, axis=-1)
    att_BxHxLxL = att_BxHxLxL.astype(cfg.dtype)
    out_BxLxHxDh = jnp.einsum('...hqk,...khd->...qhd', att_BxHxLxL, v_BxLxHxDh)
    return nn.DenseGeneral(
        features=cfg.D,
        name='attn_out_proj',
        axis=(-2, -1),
        kernel_init=fsdp.init('attn_out_proj', cfg),
        use_bias=False,
        dtype=cfg.dtype,
    )(out_BxLxHxDh)


class GistTBlock(nn.Module):
  """Pre-LN transformer block threading the attention bias."""

  docfg: DoConfig

  @nn.compact
  def __call__(
      self,
      in_BxLxD: jax.Array,
      bias_Bx1xLxL: Optional[jax.Array] = None,
  ):
    cfg = self.docfg
    x_BxLxD = nn.LayerNorm(dtype=cfg.dtype, use_bias=False)(in_BxLxD)
    # Named to match `gpt2.py`'s auto-generated `CausalAttn_0` so the parameter
    # tree is identical to the stock model: same init RNG paths (hence the same
    # weights from the same seed) and checkpoints interchangeable between the
    # baseline and the ablation.
    x_BxLxD = GistAttn(cfg, name='CausalAttn_0')(x_BxLxD, bias_Bx1xLxL)
    x_BxLxD += in_BxLxD

    z_BxLxD = nn.LayerNorm(dtype=cfg.dtype, use_bias=False)(x_BxLxD)
    z_BxLxD = Mlp(cfg)(z_BxLxD)
    return x_BxLxD + z_BxLxD


class TransformerGist(nn.Module):
  """Decoder-only transformer whose attention is gist-masked."""

  docfg: DoConfig

  def setup(self):
    cfg = self.docfg
    self.embed = nn.Embed(
        num_embeddings=cfg.V,
        features=cfg.D,
        embedding_init=fsdp.init('embedding', cfg),
    )
    self.pos_embed = nn.Embed(
        num_embeddings=cfg.L,
        features=cfg.D,
        embedding_init=fsdp.init('embedding', cfg),
    )
    block = nn.remat(GistTBlock) if cfg.remat else GistTBlock
    self.blocks = [block(cfg) for _ in range(cfg.N)]
    self.out_ln = nn.LayerNorm(dtype=cfg.dtype, use_bias=False)

  def __call__(
      self,
      y_BxL: jax.Array,
      bias_Bx1xLxL: Optional[jax.Array] = None,
  ):
    y_BxLxD = self.embed(y_BxL)
    y_BxLxD += self.pos_embed(jnp.arange(0, y_BxL.shape[1])[None, ...])
    for block in self.blocks:
      y_BxLxD = block(y_BxLxD, bias_Bx1xLxL)
    y_BxLxD = self.out_ln(y_BxLxD)
    return self.embed.attend(y_BxLxD.astype(jnp.float32))
