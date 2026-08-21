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
"""Factory for producing experimental models."""

# pylint: disable=invalid-name,g-import-not-at-top,unused-import

import functools
from typing import TYPE_CHECKING

from flax import linen as nn
from tg.models import gpt2 as model
from tg.models import gpt2_gist
from tg.models import tg_config
from tg.models import tg_model
from tg.training import gist_loss
from tg.training import loss as loss_lib
from tg.training import tg_loss

if TYPE_CHECKING:
  import ml_collections


def get_model_and_loss(
    c: "ml_collections.ConfigDict",
    vocab_size: int,
) -> tuple[nn.Module, loss_lib.LossFnFactory]:
  """Returns an instantiated (potentially experimental) model."""

  model_type = c.get("model_type", "do")

  if model_type == "do_gist":
    # GPT-2 + gist masking (paper 3.3): stock architecture, but attention is
    # restricted to the current sentence plus earlier sentences' [EOS] gists.
    cfg = model.DoConfig(**c.model, V=vocab_size)
    module = gpt2_gist.TransformerGist(cfg)
    get_loss_fn = functools.partial(
        gist_loss.get_gist_loss_fn,
        eos_id=int(c.gist_eos_id),
        special_ids=tuple(c.get("gist_special_ids", ())),
    )
    return module, get_loss_fn

  if model_type == "tg":
    # Thought Gestalt: recurrent sentence-level transformer. Its loss factory
    # takes a `tg_loss.TgBatch` (a batch of documents) rather than `in_BxL`.
    cfg = tg_config.TgConfig(**c.model, V=vocab_size)
    module = tg_model.ThoughtGestaltDo(cfg)
    get_loss_fn = functools.partial(tg_loss.get_tg_loss_fn, cfg=cfg)
    return module, get_loss_fn

  # default model and configs
  m = model
  get_loss_fn = loss_lib.get_default_loss_fn

  cfg = m.DoConfig(**c.model, V=vocab_size)  # pytype:disable=attribute-error
  module = m.TransformerDo(cfg)  # pytype:disable=attribute-error
  return module, get_loss_fn
