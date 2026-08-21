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
"""GPT-2 + gist masking baseline (paper 3.3, Table 5).

Matches TG's attention connectivity -- causal within a sentence, earlier
sentences reachable only via their `[EOS]` gist -- while removing recurrence:
gists form in the same forward pass as the tokens that read them.

The mask is derived from the paper's description rather than from the original
implementation. Its numbers are not expected to match published ones.

Usage:
  python -m tg.main --config=tg/configs/gpt2_gist.py \
      --config.tg_flat_train_path=/path/wiki.train \
      --config.tg_flat_val_path=/path/wiki.valid \
      --config.workdir=/path/runs/gpt2_gist_12M
"""

import ml_collections

from tg.configs import gpt2_boundary


def get_config() -> ml_collections.ConfigDict:
  """Gist-masking baseline: boundary-bias data plus the gist attention mask."""
  # Same flattened, boundary-marked stream as the boundary-bias baseline; the
  # only difference between the two runs is the attention mask.
  cfg = gpt2_boundary.get_config()
  cfg.model_type = "do_gist"

  # Defaults correspond to GPT-2 + [PAD] [BOS] [EOS] [EOD] = 50257..50260.
  cfg.gist_eos_id = 50259
  cfg.gist_special_ids = (50257, 50258, 50259, 50260)
  return cfg
