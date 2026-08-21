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
"""GPT-2 + sentence-boundary bias baseline (paper 3.3, Table 3).

The baseline decoder trained on TG's sentence stream flattened into continuous
tokens, keeping `[BOS]`/`[EOS]` and dropping `[PAD]`. Tests whether a plain
decoder benefits merely from being told where sentences end.

Usage:
  python -m tg.main --config=tg/configs/gpt2_boundary.py \
      --config.tg_flat_train_path=/path/wiki.train \
      --config.tg_flat_val_path=/path/wiki.valid \
      --config.workdir=/path/runs/gpt2_boundary_12M
"""

import ml_collections


def get_config() -> ml_collections.ConfigDict:
  """Boundary-bias baseline configuration."""
  cfg = ml_collections.ConfigDict()
  cfg.seed = 42

  # Stock GPT-2 -- the whole point is that the model is unchanged.
  cfg.model_type = "do"

  # Data: local flattened TG stream instead of tfds.
  cfg.ds_source = "tg_flat"
  cfg.tg_flat_train_path = ""  # required
  cfg.tg_flat_val_path = ""  # required
  cfg.tg_document_format = "wikitext"  # title-scan article splitting
  cfg.ds_name = ""  # unused under ds_source="tg_flat"
  cfg.vocab_path = ""  # unused; the GPT-2 tokenizer is built directly

  cfg.batch_size = 16
  cfg.train_epochs = 50

  # Matched to the reference GPT-2 baseline: 12 layers, d_model 768, 1024 ctx.
  cfg.model = ml_collections.config_dict.create(
      D=768,
      H=12,
      L=1024,
      N=12,
      F=3072,
      dtype="float32",
      fsdp_enabled=True,
      remat=False,
  )

  # Optimizer: the TG runs' schedule, so the comparison isolates the data.
  cfg.opt = ml_collections.config_dict.create(
      num_train_steps=31_200,
      peak_learning_rate=2.5e-4,
      init_learning_rate=0.0,
      final_learning_rate=2.5e-5,
      warmup_steps=124,
      decay_type="cosine",
      weight_decay=0.01,
      b1=0.9,
      b2=0.999,
      eps=1e-8,
      clip_by_global_norm=1.0,
      optimizer="adamw",
  )

  cfg.workdir = ""
  cfg.checkpoint = True
  cfg.checkpoint_restore_dir = None
  cfg.max_to_keep = 3
  cfg.save_every_steps = 624  # once per epoch
  cfg.eval_every_steps = 624
  cfg.eval_split = "validation"
  cfg.eval_steps = 100
  cfg.eval_max_target_length = 1024
  cfg.write_train_metrics_every_steps = 50
  cfg.write_perf_metrics_every_steps = 100
  cfg.pygrain_worker_count = 0  # unused under ds_source="tg_flat"
  cfg.pygrain_worker_buffer_size = 1
  cfg.n_data_shards = 1
  cfg.n_fsdp_shards = 1
  return cfg
