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
"""TG training loop: epoch-driven, with the chunk-size curriculum.

The loop is epoch-driven rather than step-driven because the reference is:
`_rebuild_train_for_chunk` (train_loop.py L186-320) re-chunks the corpus at
curriculum boundaries and then recomputes, from the NEW loader length,

    loader_steps_per_epoch = len(train_loader)
    schedule_step_budget   = loader_steps_per_epoch * epochs_lr
    warmup_steps           = floor(budget * warmup_ratio)

so the cosine denominator and the warmup length both move when the chunk size
moves. Bigger chunks mean fewer, longer streams, so at a fixed token budget
both `steps_per_epoch` AND the padded batch size `B` change per phase; each is
re-measured rather than assumed.

Three things force a re-jit, all at epoch boundaries and all bounded:
  * padded shape `(B, T)` changes when the chunk size changes;
  * the dropout warm-in multiplier changes at steps 2000 / 7000;
  * both are held static so the rates stay compile-time constants.
"""

# pylint: disable=invalid-name,g-importing-member,g-import-not-at-top

import dataclasses
import functools
import os
from typing import Any, TYPE_CHECKING

from absl import logging
from clu import metric_writers
from flax import linen as nn
from flax.training.train_state import TrainState
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np
import optax
import orbax.checkpoint as ocp

from tg.data import tg as tg_data
from tg.models import factory as model_factory
from tg.models import tg_config
from tg.training import tg_loss
from tg.training import tg_optimizer

if TYPE_CHECKING:
  import ml_collections

PyTree = Any


# --------------------------------------------------------------------------- #
# Curriculum
# --------------------------------------------------------------------------- #
def chunk_size_for_epoch(epoch: int, c: "ml_collections.ConfigDict") -> int:
  """`chunk_cl_start + step * (epoch // interval)`, UNCAPPED.

  Closed form of the reference's incremental schedule. `train_loop.py` L523-525
  keeps `cl_state["current_chunk"]` and adds `step` whenever
  `epoch >= next_epoch_trigger`, with `next_epoch_trigger` initialized to
  `interval` (`data_setup.py` L375) and advanced by `interval` on each fire.
  That fires at epochs 5, 10, 15, ... so the chunk size at epoch e is exactly
  `start + step * (e // interval)`.

  There is no ceiling: the reference never clamps `new_chunk`, and
  `--chunk_size` is overwritten with `chunk_cl_start` at parse time
  (`args_tg.py` L470) whenever the curriculum is enabled, so it acts as the
  fixed size only when the curriculum is OFF. The schedule therefore runs
  30 -> 42 -> 54 -> 66 -> 78 -> ..., which is what reaches the ~2000-token
  dependency span described in paper section 2.3.
  """
  if not c.get('chunk_cl_enable', True):
    return int(c.chunk_size)
  start = int(c.get('chunk_cl_start', 30))
  step = int(c.get('chunk_cl_step', 12))
  interval = max(1, int(c.get('chunk_cl_interval_epochs', 5)))
  return int(start + step * (int(epoch) // interval))


def eos_weight_for_epoch(epoch: int, c: "ml_collections.ConfigDict") -> float:
  """`freq_token_schedule="1"` => full weight in epoch 0, target thereafter.

  `data_setup.py` L411-427 turns the schedule into
  `schedule_weights = (1.0, target)` with a cutoff at `1 * steps_per_epoch`;
  expressed per-epoch it is exactly this.
  """
  if not c.get('eos_loss_weight_enable', True):
    return 1.0
  boundary = int(c.get('eos_loss_schedule_epochs', 1))
  return 1.0 if int(epoch) < boundary else float(c.eos_loss_weight)


# --------------------------------------------------------------------------- #
# Phase (one curriculum setting)
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Phase:
  chunk_size: int
  streams: list
  sampler: Any
  pad_B: int
  pad_T: int
  steps_per_epoch: int


def build_phase(
    chunk_size: int,
    train_docs,
    val_docs,
    specials,
    cfg,
    c: "ml_collections.ConfigDict",
    n_devices: int,
) -> Phase:
  """Re-chunk, re-sample and re-measure everything that depends on chunk size."""
  streams = tg_data.streams_from_cached_corpus(
      train_docs, specials, cfg, chunk_size,
      min_sentences_per_document=c.min_sentences_per_document)
  sampler = tg_data.make_token_budget_sampler(
      streams,
      target_tokens_per_batch=c.target_tokens_per_batch,
      bucket_width=c.bucket_width,
      batch_size=c.batch_size,
      seed=c.seed,
  )
  pad_B = int(np.ceil(sampler.max_examples_per_batch / n_devices) * n_devices)
  pad_T = int(chunk_size)

  # `loader_steps_per_epoch = len(train_loader)`: the ACTUAL batch count, not
  # the estimate, because the LR budget is derived from it.
  sampler.set_epoch(0)
  steps_per_epoch = sum(1 for _ in sampler)

  logging.info(
      'CL phase chunk_size=%d: train %d streams -> %d steps/epoch | '
      'padded (B=%d, T=%d)',
      chunk_size, len(streams), steps_per_epoch, pad_B, pad_T)
  return Phase(chunk_size, streams, sampler, pad_B, pad_T, steps_per_epoch)


# --------------------------------------------------------------------------- #
# Model / step
# --------------------------------------------------------------------------- #
def make_module(cfg, dropout_scale: float):
  """A module whose token/memory/srep dropout carries the warm-in multiplier."""
  from tg.models import tg_model
  phase_cfg = dataclasses.replace(cfg, dropout_scale=float(dropout_scale))
  return tg_model.ThoughtGestaltDo(phase_cfg), phase_cfg


def _train_step(state, batch, dropout_rng, lr, eos_weight, c, cfg, mesh,
                apply_fn):
  """One optimizer update over a whole document batch."""
  if mesh is not None:
    shard = lambda x: jax.lax.with_sharding_constraint(
        x, NamedSharding(mesh, P('data')))
    batch = tg_loss.TgBatch(shard(batch.sentences_BxTxL),
                            shard(batch.masks_BxTxL), shard(batch.lengths_B))

  loss_fn = tg_loss.get_tg_loss_fn(
      batch, apply_fn, c, cfg=cfg, dropout_rng=dropout_rng,
      eos_weight=eos_weight)
  (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

  opt_state = tg_optimizer.set_learning_rate(state.opt_state, lr)
  updates, new_opt_state = state.tx.update(grads, opt_state, state.params)
  new_params = optax.apply_updates(state.params, updates)
  new_state = state.replace(step=state.step + 1, params=new_params,
                            opt_state=new_opt_state)
  gnorm = optax.global_norm(grads)
  metrics = {
      'train_loss': loss,
      'log_perplexity': aux.log_perplexity,
      'train_ntokens': aux.ntokens,
      'learning_rate': lr,
      'grad_norm': gnorm,
      'eos_weight': eos_weight,
  }
  return new_state, metrics


def build_val_streams(val_docs, specials, cfg, c):
  """Validation streams, UNCHUNKED -- one stream per document.

  The reference run has `val_use_chunking=False`, so the STM runs continuously
  across a whole article instead of being reset every `chunk_size` sentences.
  Chunking validation was measurably pessimistic: on the same checkpoint it
  cost 0.014 nats (ppl 54.78 chunked@48 vs 54.02 unchunked), because most val
  documents are far longer than a chunk (median 119 sentences, max 706).
  """
  out = []
  for i, sentences in enumerate(val_docs):
    if len(sentences) < c.min_sentences_per_document:
      continue
    ids, mask = tg_data.tensorize_document(
        sentences, specials, max_sentence_tokens=cfg.max_sentence_tokens,
        sentence_tail_len=cfg.sentence_tail_len)
    out.append(tg_data.Stream(ids, mask, f'val_{i}'))
  return out


def make_sentence_eval_fn(mod, cfg):
  """Jitted SINGLE-sentence forward + CE, for the unchunked eval loop.

  Documents reach 706 sentences, so the sentence loop cannot be unrolled into
  one graph for eval. Shapes here are fixed (B=1), so this compiles once and is
  reused across every phase.
  """
  from optax import losses as _losses
  from tg.models import tg_model as _tgm

  @jax.jit
  def fn(params, ids_1xL, mask_1xL, kv, valid, bos_ctx, bos_valid, step_ids):
    out = mod.apply({'params': params}, ids_1xL, mask_1xL, kv, valid, bos_ctx,
                    bos_valid, True, step_ids)
    logits = out.logits_BxLxV[:, :-1, :]
    labels = ids_1xL[:, 1:]
    ok = ((mask_1xL[:, 1:] == 1) & (mask_1xL[:, :-1] == 1)).astype(jnp.float32)
    ce = _losses.softmax_cross_entropy_with_integer_labels(logits, labels)
    return jnp.sum(ce * ok), jnp.sum(ok), out.srep_BxD, out.has_eos_B

  del _tgm
  return fn


def evaluate_unchunked(sent_eval_fn, params, val_streams, cfg) -> dict:
  """Mean CE over validation with the STM running for a whole document."""
  from tg.models import tg_model
  ce_sum, ntok = 0.0, 0.0
  for s in val_streams:
    mem = tg_model.init_memory(1, cfg)
    bos_ctx = jnp.zeros((1, cfg.D), cfg.dtype)
    bos_valid = jnp.zeros((1,), jnp.bool_)
    ids = jnp.asarray(s.ids_SxL)
    mask = jnp.asarray(s.mask_SxL)
    n = s.num_sentences
    for t in range(n):
      c_, n_, srep, has_eos = sent_eval_fn(
          params, ids[t:t + 1], mask[t:t + 1], mem.kv_BxMxD, mem.valid_BxM,
          bos_ctx, bos_valid, mem.step_BxM)
      ce_sum += float(c_)
      ntok += float(n_)
      # `step=t` only matters for multi_eos_pos_mode='grouped'; eval is
      # deterministic so no backprop window is needed.
      mem = tg_model.push_memory(mem, srep, has_eos, step=t)
      bos_ctx = srep
      bos_valid = has_eos & jnp.asarray([t + 1 < n])
  loss = ce_sum / max(ntok, 1.0)
  return {'eval_loss': loss, 'eval_perplexity': float(np.exp(loss)),
          'eval_ntokens': ntok}


def init_state(c, module, rng, mesh, tx):
  cfg = module.cfg
  args = (
      jax.ShapeDtypeStruct((1, cfg.L), jnp.int32),
      jax.ShapeDtypeStruct((1, cfg.L), jnp.int32),
      jax.ShapeDtypeStruct((1, cfg.M, cfg.D), cfg.dtype),
      jax.ShapeDtypeStruct((1, cfg.M), jnp.bool_),
      jax.ShapeDtypeStruct((1, cfg.D), cfg.dtype),
      jax.ShapeDtypeStruct((1,), jnp.bool_),
  )

  def init(rng, *inputs):
    params = module.init(rng, *inputs, True)
    return TrainState.create(apply_fn=module.apply, params=params['params'],
                             tx=tx)

  params = jax.eval_shape(init, rng, *args)
  shardings = nn.get_sharding(params, mesh)
  state = jax.jit(init, out_shardings=shardings)(rng, *args)
  return shardings, state


def ckpt_manager(ckpt_dir, c):
  return ocp.CheckpointManager(
      ckpt_dir,
      options=ocp.CheckpointManagerOptions(
          max_to_keep=c.max_to_keep, save_interval_steps=1,
          best_fn=lambda m: float(m['eval_loss']), best_mode='min'))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def train_and_evaluate(c: "ml_collections.ConfigDict"):
  mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), ('data',))
  n_devices = jax.device_count()
  os.makedirs(c.workdir, exist_ok=True)
  rng = jax.random.PRNGKey(c.seed)
  rng, dropout_rng = jax.random.split(rng)

  train_docs, train_meta = tg_data.load_cached_corpus(c.tg_train_cache)
  val_docs, _ = tg_data.load_cached_corpus(c.tg_val_cache)
  logging.info('TG cache: %d train docs (%d tokens, splitter=%s), %d val docs',
               len(train_docs), train_meta.get('tokens'),
               train_meta.get('splitter_model_name'), len(val_docs))

  vocab_size = int(train_meta['vocab_size'])
  specials = tg_data.SpecialIds(
      bos=tg_config.DEFAULT_BOS_ID, eos=tg_config.DEFAULT_EOS_ID,
      eod=tg_config.DEFAULT_EOD_ID, pad=tg_config.DEFAULT_PAD_ID)
  with c.unlocked():
    c.model.bos_id, c.model.eos_id = specials.bos, specials.eos
    c.model.eod_id, c.model.pad_id = specials.eod, specials.pad

  base_module, _ = model_factory.get_model_and_loss(c, vocab_size)
  cfg = base_module.cfg

  tx = tg_optimizer.make_tg_optimizer(
      peak_learning_rate=c.opt.peak_learning_rate,
      weight_decay=c.opt.weight_decay,
      b1=c.opt.get('b1', 0.9), b2=c.opt.get('b2', 0.999),
      clip_by_global_norm=c.opt.get('clip_by_global_norm', 1.0))

  total_epochs = int(c.train_epochs)
  warmup_ratio = float(c.opt.warmup_ratio)
  min_scale = float(c.opt.cosine_min_lr_scale)

  # Validation is UNCHUNKED and chunk-independent, so it is built once.
  val_streams = build_val_streams(val_docs, specials, cfg, c)
  logging.info('TG val: %d streams (UNCHUNKED, %d sentences) '
               '-- val_use_chunking=False, matching the reference',
               len(val_streams), sum(s_.num_sentences for s_ in val_streams))

  # Build the first phase to size the model init.
  phase = build_phase(chunk_size_for_epoch(0, c), train_docs, val_docs,
                      specials, cfg, c, n_devices)
  module, phase_cfg = make_module(cfg, 0.0)
  shardings, state = init_state(c, module, rng, mesh, tx)

  mngr = ckpt_manager(os.path.join(c.workdir, 'checkpoints'), c) \
      if c.checkpoint else None
  if mngr is not None and mngr.latest_step() is not None:
    logging.info('Restoring TG checkpoint %d', mngr.latest_step())
    state = mngr.restore(mngr.latest_step(),
                         args=ocp.args.StandardRestore(state))

  writer = metric_writers.create_default_writer(
      c.workdir, just_logging=jax.process_index() > 0)
  writer.write_hparams(dict(c))

  global_step = int(state.step)
  budget = phase.steps_per_epoch * total_epochs
  cur_key = None
  step_fn = None

  eval_mod, eval_cfg = make_module(cfg, 0.0)   # eval never uses dropout
  sent_eval_fn = make_sentence_eval_fn(eval_mod, eval_cfg)

  def rebuild(dropout_scale, ph):
    nonlocal step_fn
    mod, pcfg = make_module(cfg, dropout_scale)
    step_fn = jax.jit(
        functools.partial(_train_step, c=c, cfg=pcfg, mesh=mesh,
                          apply_fn=mod.apply),
        donate_argnames=('state',))
    logging.info('re-jit: dropout_scale=%.1f chunk=%d (B=%d,T=%d)',
                 dropout_scale, ph.chunk_size, ph.pad_B, ph.pad_T)

  def _eval_and_checkpoint(epoch, when):
    """Unchunked validation + best-val checkpoint. Called 2x per epoch."""
    m = evaluate_unchunked(sent_eval_fn, state.params, val_streams, cfg)
    m['epoch'] = epoch + 1
    m['chunk_size'] = phase.chunk_size
    try:  # real peak HBM, not the 75% XLA preallocation nvidia-smi reports
      ms = jax.local_devices()[0].memory_stats() or {}
      m['peak_gib'] = float(ms.get('peak_bytes_in_use', 0)) / 2**30
      m['bytes_limit_gib'] = float(ms.get('bytes_limit', 0)) / 2**30
    except Exception:  # pylint: disable=broad-except
      pass
    writer.write_scalars(global_step, m)
    logging.info(
        'epoch %d/%d (%s) step %d: eval_loss=%.4f ppl=%.2f '
        '(chunk=%d, B=%d, T=%d, peak=%.1f/%.1f GiB)',
        epoch + 1, total_epochs, when, global_step, m['eval_loss'],
        m['eval_perplexity'], phase.chunk_size, phase.pad_B, phase.pad_T,
        m.get('peak_gib', 0.0), m.get('bytes_limit_gib', 0.0))
    if mngr is not None:
      mngr.save(global_step, args=ocp.args.StandardSave(state),
                metrics={'eval_loss': float(m['eval_loss'])})

  with metric_writers.ensure_flushes(writer):
    for epoch in range(total_epochs):
      new_chunk = chunk_size_for_epoch(epoch, c)
      if new_chunk != phase.chunk_size:
        phase = build_phase(new_chunk, train_docs, val_docs, specials, cfg, c,
                            n_devices)
        # Mirror `_rebuild_train_for_chunk`: budget and warmup follow the new
        # steps/epoch, and the eval cadence is one epoch by construction.
        budget = phase.steps_per_epoch * total_epochs
        logging.info(
            'Chunk-size CL rebuild (epoch_%d): chunk_size=%d | '
            'steps_per_epoch=%d | schedule_step_budget=%d | warmup_steps=%d',
            epoch, new_chunk, phase.steps_per_epoch, budget,
            tg_optimizer.warmup_steps_for(budget, warmup_ratio))

      eos_w = eos_weight_for_epoch(epoch, c)
      evals_per_epoch = int(c.get('evals_per_epoch', 2))
      max_steps = int(c.get('max_steps_per_epoch', 0) or 0)
      # Mid-point must follow the EFFECTIVE epoch length, otherwise a capped
      # (debug/probe) epoch never reaches it.
      effective_steps = (min(phase.steps_per_epoch, max_steps) if max_steps
                         else phase.steps_per_epoch)
      mid_point = max(1, effective_steps // 2)
      phase.sampler.set_epoch(epoch)
      steps_this_epoch = 0
      for idx in phase.sampler:
        if max_steps and steps_this_epoch >= max_steps:
          break
        steps_this_epoch += 1
        dscale = tg_config.dropout_multiplier_for_step(
            global_step, cfg.dropout_half_step, cfg.dropout_full_step)
        key = (dscale, phase.pad_B, phase.pad_T)
        if key != cur_key:
          rebuild(dscale, phase)
          cur_key = key

        sentences, masks, lengths = tg_data.make_batch(
            [phase.streams[i] for i in idx], phase.pad_T, phase.pad_B)
        batch = tg_loss.TgBatch(jnp.asarray(sentences), jnp.asarray(masks),
                                jnp.asarray(lengths))
        lr = c.opt.peak_learning_rate * tg_optimizer.lr_scale(
            global_step, budget, warmup_ratio, min_scale)
        state, metrics = step_fn(
            state, batch, jax.random.fold_in(dropout_rng, global_step),
            jnp.asarray(lr, jnp.float32), jnp.asarray(eos_w, jnp.float32))
        # Mid-epoch eval (evals_per_epoch=2, matching the reference cadence).
        if (evals_per_epoch > 1 and steps_this_epoch == mid_point):
          _eval_and_checkpoint(epoch, 'mid')
        if global_step % c.write_train_metrics_every_steps == 0:
          m = jax.device_get(metrics)
          writer.write_scalars(global_step, {k: float(v) for k, v in m.items()})
          if not np.isfinite(float(m['train_loss'])):
            raise FloatingPointError(global_step, float(m['train_loss']))
        global_step += 1

      _eval_and_checkpoint(epoch, 'end')

  if mngr is not None:
    mngr.wait_until_finished()
    logging.info('TG checkpoints latest=%s best=%s', mngr.latest_step(),
                 mngr.best_step())
    mngr.close()
  return state
