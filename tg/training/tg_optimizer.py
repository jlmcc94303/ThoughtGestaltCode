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
"""TG optimizer + LR schedule, matching the PyTorch reference.

Two things differ from `tg/training/optimizer.py` and need their own path:

* **no-decay parameter group** -- `model_setup.py:_build_param_groups` (L21-38)
  puts 1D tensors, biases, norms and gates in a `weight_decay=0.0` group.
* **runtime learning rate** -- the chunk-size curriculum rebuilds the data
  loader at epoch boundaries, and `_rebuild_train_for_chunk` recomputes
  `schedule_step_budget = loader_steps_per_epoch * epochs_lr` (and hence the
  warmup length) every time. The cosine denominator is therefore not known at
  construction, so the LR is injected per step instead of baked into a
  `optax.schedule`.
"""

# pylint: disable=invalid-name,g-importing-member

import math
import re
from typing import Any

import jax
import optax

PyTree = Any

# `_build_param_groups`: no weight decay on norms / gates / biases.
_NO_DECAY_RE = re.compile(r'(\.bias$|bias$|ln_|layernorm|\.ln|norm|gate)',
                          re.IGNORECASE)


def _path_name(path) -> str:
  parts = []
  for p in path:
    parts.append(getattr(p, 'key', None) or str(getattr(p, 'idx', p)))
  return '/'.join(str(x) for x in parts)


def decay_mask(params: PyTree) -> PyTree:
  """True where weight decay applies.

  Mirrors `_build_param_groups`: `p.ndim == 1` (which also covers every bias,
  LayerNorm scale/bias and the scalar gates), or a name that looks like a
  bias/norm/gate, goes in the no-decay group.
  """
  def _fn(path, x):
    name = _path_name(path)
    no_decay = (x.ndim < 2) or bool(_NO_DECAY_RE.search(name))
    return not no_decay

  return jax.tree_util.tree_map_with_path(_fn, params)


def lr_scale(step: int, budget: int, warmup_ratio: float,
             min_scale: float, schedule: str = 'cosine') -> float:
  """Port of `model_setup.py:build_scheduler._schedule_value`.

  Linear warmup **from 0** over `warmup_ratio` of the budget, then cosine down
  to `min_scale`. Note this differs from NanoDO's schedule, which warms up from
  a non-zero `init_learning_rate`.
  """
  budget = max(1, int(budget))
  progress = min(1.0, max(0.0, float(step) / float(budget)))
  warmup_cutoff = max(0.0, min(1.0, float(warmup_ratio)))
  if warmup_cutoff > 0.0 and progress < warmup_cutoff:
    return progress / max(1e-9, warmup_cutoff)
  decay_span = max(1e-9, 1.0 - warmup_cutoff)
  decay_progress = min(1.0, max(0.0, (progress - warmup_cutoff) / decay_span))
  if schedule == 'cosine':
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_scale + (1.0 - min_scale) * cosine
  return max(0.0, 1.0 - decay_progress)


def warmup_steps_for(budget: int, ratio: float) -> int:
  """Port of `model_setup.py:_compute_warmup_steps`."""
  ratio = max(0.0, min(1.0, float(ratio)))
  steps = int(math.floor(budget * ratio))
  if ratio > 0.0:
    steps = max(1, steps)
  return min(budget, steps)


def mup_lr_mask(params: PyTree) -> PyTree:
  """True on hidden matrices, which are the parameters muP rescales.

  muP splits parameters into three classes. Embeddings and the readout are
  "vector-like" in the input/output dimension and keep the standard learning
  rate; norms, gates and biases are 1-D and likewise unscaled. Only hidden
  matrices -- attention projections, MLP kernels, the S_REP head -- get their
  learning rate divided by the width ratio, which is what makes a rate tuned at
  the base width transfer to a wider model.
  """
  def _fn(path, x):
    name = _path_name(path).lower()
    if x.ndim < 2:
      return False  # norms, gates, biases
    if 'embed' in name or 'lm_head' in name or 'readout' in name:
      return False  # vector-like in V
    return True

  return jax.tree_util.tree_map_with_path(_fn, params)


def make_mup_scaled_optimizer(
    base: optax.GradientTransformation,
    params: PyTree,
    lr_scale: float,
) -> optax.GradientTransformation:
  """Apply muP's per-parameter learning-rate scaling on top of `base`.

  Implemented as a masked extra scaling of the *updates* rather than separate
  parameter groups, so the AdamW state layout is unchanged and a muP run can be
  compared against a standard run parameter-for-parameter.
  """
  if lr_scale == 1.0:
    return base
  mask = mup_lr_mask(params)
  scale_hidden = optax.masked(optax.scale(lr_scale), mask)
  return optax.chain(base, scale_hidden)


def make_tg_optimizer(
    peak_learning_rate: float,
    weight_decay: float = 0.01,
    b1: float = 0.9,
    b2: float = 0.999,
    clip_by_global_norm: float | None = 1.0,
) -> optax.GradientTransformation:
  """AdamW with the reference's betas, wd and no-decay mask; LR injected."""
  def _make(learning_rate):
    chain = []
    if clip_by_global_norm:
      chain.append(optax.clip_by_global_norm(clip_by_global_norm))
    chain.append(
        optax.adamw(
            learning_rate=learning_rate,
            b1=b1,
            b2=b2,
            weight_decay=weight_decay,
            mask=decay_mask,
        )
    )
    return optax.chain(*chain)

  return optax.inject_hyperparams(_make)(learning_rate=peak_learning_rate)


def set_learning_rate(opt_state, lr):
  """Return `opt_state` with the injected learning rate replaced."""
  return opt_state._replace(
      hyperparams={**opt_state.hyperparams, 'learning_rate': lr}
  )
