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
"""TG loss: multi-step next-token cross-entropy + S_REP norm regularizer.

Port of the loss section of `train_pipelines/tg/train_loop.py`:

    shift_logits = logits[:, :-1]
    shift_labels = ids[:, 1:]
    valid = ~key_pad_mask[:, 1:] & ~loss_mask_full[:, :-1]
    step_loss = mean_CE_over_valid + srep_norm_reg_loss
    total_batch_loss = sum(step_losses)   # ONE backward per document batch

`loss_mask_full` equals `key_pad_mask` in external-memory (non in-context) mode,
so a position counts only when both it and its predecessor are non-padding.
"""

# pylint: disable=invalid-name,g-importing-member,g-bare-generic

from typing import Any, Callable, Optional, TYPE_CHECKING

from flax.struct import dataclass
import jax
import jax.numpy as jnp
from tg.models import tg_model
from tg.models.tg_config import TgConfig
from tg.training import loss as loss_lib
from optax import losses

if TYPE_CHECKING:
  import ml_collections

PyTree = Any


@dataclass
class TgBatch:
  """A batch of documents, each already tensorized into sentences."""

  sentences_BxTxL: jax.Array  # token ids, zero-padded past `lengths_B`
  masks_BxTxL: jax.Array  # 1 = real token, 0 = [PAD]
  lengths_B: jax.Array  # sentences per document


@dataclass
class TgLossTerms:
  """Per-step quantities, summed over sentence steps by `run_sentence_loop`."""

  loss: jax.Array  # optimized objective: sum_t (mean CE_t + reg_t)
  main_loss: jax.Array  # sum_t mean CE_t
  norm_reg_loss: jax.Array  # sum_t reg_t
  ce_sum: jax.Array  # sum of per-token CE over all valid positions
  ntokens: jax.Array  # number of valid positions
  nsentences: jax.Array  # number of real (non-padded) sentence rows
  srep_norm_sum: jax.Array  # sum of pre-normalization S_REP norms


def sentence_loss_terms(
    out: tg_model.TgStepOutput,
    ids_BxL: jax.Array,
    mask_BxL: jax.Array,
    row_valid_B: jax.Array,
    cfg: TgConfig,
    eos_weight: Optional[jax.Array] = None,
) -> TgLossTerms:
  """Shifted CE + hinge regularizer for one sentence step.

  `eos_weight` down-weights positions whose TARGET is `[EOS]`, mirroring
  `core/loss_weighting.py:compute_weighted_loss_terms`: the reduction becomes
  `sum(w * loss) / sum(w)` rather than a plain mean, so the denominator shrinks
  with the weights. Pass a traced scalar so the schedule costs no recompile;
  `None` (or 1.0) reproduces the unweighted mean exactly.
  """
  logits_BxLmxV = out.logits_BxLxV[:, :-1, :]
  labels_BxLm = ids_BxL[:, 1:]
  valid_BxLm = (
      (mask_BxL[:, 1:] == 1)
      & (mask_BxL[:, :-1] == 1)
      & row_valid_B[:, None]
  )
  valid_f_BxLm = valid_BxLm.astype(jnp.float32)

  ce_BxLm = losses.softmax_cross_entropy_with_integer_labels(
      logits_BxLmxV, labels_BxLm
  )
  ce_sum = jnp.sum(ce_BxLm * valid_f_BxLm)
  ntokens = jnp.sum(valid_f_BxLm)

  # Which label positions count as EOS. Under multi-EOS the reference's
  # `mask_all` mode treats every EOS id as an EOS target, while `last_only`
  # supervises just the final one -- so only that id is down-weighted.
  eos_ids = tuple(cfg.multi_eos_ids or (cfg.eos_id,))
  if cfg.multi_eos_enabled and cfg.multi_eos_loss_mode == 'last_only':
    eos_ids = (eos_ids[-1],)

  if eos_weight is None:
    main_loss = ce_sum / jnp.maximum(ntokens, 1.0)
  else:
    # weight 1.0 everywhere except EOS targets, which get `eos_weight`.
    is_eos_BxLm = jnp.zeros_like(labels_BxLm, dtype=jnp.bool_)
    for tok in eos_ids:
      is_eos_BxLm = is_eos_BxLm | (labels_BxLm == int(tok))
    w_BxLm = jnp.where(is_eos_BxLm, eos_weight, 1.0) * valid_f_BxLm
    weighted_sum = jnp.sum(ce_BxLm * w_BxLm)
    weight_total = jnp.sum(w_BxLm)
    main_loss = weighted_sum / jnp.maximum(weight_total, 1e-6)

  row_valid_f_B = row_valid_B.astype(jnp.float32)
  nsentences = jnp.sum(row_valid_f_B)
  penalty_sum = jnp.sum(out.srep_norm_penalty_B * row_valid_f_B)
  norm_reg_loss = (
      penalty_sum / jnp.maximum(nsentences, 1.0) * cfg.srep_norm_reg_weight
  )

  return TgLossTerms(
      loss=main_loss + norm_reg_loss,
      main_loss=main_loss,
      norm_reg_loss=norm_reg_loss,
      ce_sum=ce_sum,
      ntokens=ntokens,
      nsentences=nsentences,
      srep_norm_sum=jnp.sum(out.srep_raw_norm_B * row_valid_f_B),
  )


def tg_document_loss(
    params: PyTree,
    apply_fn: Callable,
    batch: TgBatch,
    cfg: TgConfig,
    *,
    deterministic: bool = True,
    dropout_rng: Optional[jax.Array] = None,
    eos_weight: Optional[jax.Array] = None,
) -> TgLossTerms:
  """Run every sentence step of `batch` and accumulate the loss terms."""
  step_fn = lambda t, out, ids, mask, row_valid: sentence_loss_terms(
      out, ids, mask, row_valid, cfg, eos_weight
  )
  return tg_model.run_sentence_loop(
      apply_fn,
      {'params': params},
      batch.sentences_BxTxL,
      batch.masks_BxTxL,
      batch.lengths_B,
      cfg,
      step_fn=step_fn,
      deterministic=deterministic,
      dropout_rng=dropout_rng,
  )


def get_tg_loss_fn(
    batch: TgBatch,
    apply_fn: Callable,
    c: "ml_collections.ConfigDict",
    *,
    cfg: TgConfig,
    dropout_rng: Optional[jax.Array] = None,
    eos_weight: Optional[jax.Array] = None,
) -> loss_lib.LossFn:
  """`LossFnFactory` for TG, for use with `jax.value_and_grad`."""
  del c
  deterministic = dropout_rng is None

  def loss_fn(params: PyTree) -> tuple[jax.Array, loss_lib.LossAuxData]:
    terms = tg_document_loss(
        params,
        apply_fn,
        batch,
        cfg,
        deterministic=deterministic,
        dropout_rng=dropout_rng,
        eos_weight=eos_weight,
    )
    ntokens = jnp.maximum(terms.ntokens, 1.0)
    # Reported per-token CE; the *optimized* objective is `terms.loss`, a sum
    # over sentence steps (one optax update per document batch).
    log_perplexity = terms.ce_sum / ntokens
    return terms.loss, loss_lib.LossAuxData(
        ntokens=terms.ntokens, state=(), log_perplexity=log_perplexity
    )

  return loss_fn
