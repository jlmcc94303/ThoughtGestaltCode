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
"""Gist attention mask for the GPT-2 + gist-masking baseline (paper 3.3).

A token attends causally within its own sentence, and reaches earlier
sentences only through each one's `[EOS]` gist:

    allow[q, k] = (k <= q)                        # causal
                  AND ( chunk[k] == chunk[q]      # same sentence, or
                        OR (is_eos[k] AND chunk[k] < chunk[q]) )  # prior gist

`[EOS]` belongs to the sentence it terminates, so it is that chunk's last token
and a visible gist for every later chunk.

Derived from the paper, NOT ported: the reference's `gist/bias.py` indexes the
gist flag on the query axis instead of the key axis, which grants zero
cross-sentence access. Expect our numbers to differ from published ones.
"""

# pylint: disable=invalid-name

from typing import Optional

import jax.numpy as jnp


def chunk_ids(is_eos_BxL: jnp.ndarray) -> jnp.ndarray:
  """Sentence index per position, with `[EOS]` staying in its own sentence."""
  eos = is_eos_BxL.astype(jnp.int32)
  return jnp.cumsum(eos, axis=1) - eos


def gist_allow_mask(
    input_ids_BxL: jnp.ndarray,
    eos_id: int,
    attention_mask_BxL: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
  """Boolean `[B, 1, L, L]`; True where attention is permitted."""
  is_eos_BxL = input_ids_BxL == eos_id
  chunk_BxL = chunk_ids(is_eos_BxL)

  L = input_ids_BxL.shape[1]
  causal_LxL = jnp.tril(jnp.ones((L, L), dtype=jnp.bool_))

  # Axis convention: dim 2 is the QUERY, dim 3 is the KEY.
  q_chunk = chunk_BxL[:, None, :, None]  # [B, 1, L, 1]
  k_chunk = chunk_BxL[:, None, None, :]  # [B, 1, 1, L]
  k_is_eos = is_eos_BxL[:, None, None, :]  # gist flag on the KEY

  same_sentence = q_chunk == k_chunk
  earlier_gist = k_is_eos & (k_chunk < q_chunk)

  allow = (same_sentence | earlier_gist) & causal_LxL[None, None]
  if attention_mask_BxL is not None:
    valid = attention_mask_BxL.astype(jnp.bool_)
    allow = allow & valid[:, None, None, :]  # key is real
    allow = allow & valid[:, None, :, None]  # query is real
  return allow


def gist_attention_bias(
    input_ids_BxL: jnp.ndarray,
    eos_id: int,
    attention_mask_BxL: Optional[jnp.ndarray] = None,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
  """Additive `[B, 1, L, L]` bias: 0 where allowed, large negative elsewhere."""
  allow = gist_allow_mask(input_ids_BxL, eos_id, attention_mask_BxL)
  neg = jnp.finfo(dtype).min
  return jnp.where(allow, jnp.zeros((), dtype), jnp.full((), neg, dtype))
