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
"""Loss for the GPT-2 + gist-masking baseline (paper 3.3, Table 5).

`loss.get_default_loss_fn` plus the gist attention bias, which is built from
the batch's own token ids inside the loss (it is data-dependent, and one
boolean reduction per step is negligible next to the attention it gates).

Reported perplexity excludes special-token label positions, so it is comparable
with TG's `*_no_special` metric.
"""

# pylint: disable=invalid-name,g-importing-member

from typing import Any, Callable, Optional, Sequence, TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax

from tg.data import gist as gist_lib
from tg.data import gpt2 as data
from tg.training import loss as loss_lib

if TYPE_CHECKING:
  import ml_collections

PyTree = Any


def lexical_weights(
    y_BxL: jax.Array,
    base_weights_BxL: jax.Array,
    special_ids: Sequence[int],
) -> jax.Array:
  """Zero out label positions whose target is a special token."""
  keep = jnp.ones_like(base_weights_BxL, dtype=jnp.bool_)
  for tok in special_ids:
    keep = keep & (y_BxL != int(tok))
  return base_weights_BxL * keep.astype(base_weights_BxL.dtype)


def get_gist_loss_fn(
    in_BxL: jax.Array,
    apply_fn: Callable,
    c: "ml_collections.ConfigDict",
    *,
    eos_id: Optional[int] = None,
    special_ids: Sequence[int] = (),
) -> loss_lib.LossFn:
  """`LossFnFactory` for the gist-masked baseline."""
  if eos_id is None:
    eos_id = int(c.gist_eos_id)
  special_ids = tuple(special_ids) or tuple(c.get("gist_special_ids", ()))

  def loss_fn(params: PyTree) -> tuple[jax.Array, loss_lib.LossAuxData]:
    x_BxL, y_BxL, weights_BxL = data.get_in_out(in_BxL)

    # Sentence structure is read off the inputs, so the mask always matches the
    # stream the model is actually being shown.
    bias = gist_lib.gist_attention_bias(x_BxL, eos_id)

    mutable = (
        "intermediate_acts",) if c.get("log_internal_metrics", False) else ()
    logits_BxLxV, state = apply_fn(
        {"params": params},
        x_BxL,
        bias,
        mutable=mutable,
    )

    losses_BxL = optax.softmax_cross_entropy_with_integer_labels(
        logits_BxLxV, y_BxL
    )
    # Trained on every position, including boundary tokens...
    ntokens = weights_BxL.sum()
    mean_loss = jnp.sum(losses_BxL * weights_BxL) / ntokens

    # ...but reported on lexical positions only, matching TG's protocol.
    if special_ids:
      lex_w = lexical_weights(y_BxL, weights_BxL, special_ids)
      lex_n = jnp.maximum(lex_w.sum(), 1.0)
      log_perplexity = jnp.sum(losses_BxL * lex_w) / lex_n
    else:
      log_perplexity = mean_loss

    return mean_loss, loss_lib.LossAuxData(
        ntokens=ntokens, state=state, log_perplexity=log_perplexity)

  return loss_fn
