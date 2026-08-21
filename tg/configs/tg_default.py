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
"""Thought Gestalt (TG) hyperparameters."""

import ml_collections


def get_config() -> ml_collections.ConfigDict:
  """Get the TG hyperparameter configuration."""
  cfg = ml_collections.ConfigDict()
  cfg.seed = 42
  cfg.model_type = "tg"

  # With bucket_sampler, batches are formed by TOKEN BUDGET rather than
  # document count; batch_size only seeds the sampler's fallback arithmetic.
  cfg.batch_size = 16
  cfg.bucket_sampler = True
  cfg.target_tokens_per_batch = 20000
  cfg.bucket_width = 20
  cfg.train_epochs = 50

  # Streams grow 30 -> 42 -> 54 -> 66 ... with NO ceiling.
  cfg.use_chunking = True
  cfg.chunk_size = 48  # only used when chunk_cl_enable is False
  cfg.chunk_cl_enable = True
  cfg.chunk_cl_start = 30
  cfg.chunk_cl_step = 12
  cfg.chunk_cl_interval_epochs = 5

  # EOS down-weighting: full weight during epoch 0, then 0.05.
  cfg.eos_loss_weight_enable = True
  cfg.eos_loss_weight = 0.05
  cfg.eos_loss_schedule_epochs = 1
  cfg.tg_data_path = ""  # training corpus: .txt / .jsonl
  cfg.tg_val_data_path = ""  # held-out corpus for best-checkpoint selection
  # Preprocessed caches (SaT-split + tokenized); consumed in preference to the
  # raw paths above, which use the regex splitter.
  cfg.tg_train_cache = ""
  cfg.tg_val_cache = ""
  # "wikitext" splits at "= Title =", "paragraph" on blank lines. Memory is
  # reset per document, so this must match real article boundaries.
  cfg.tg_document_format = "wikitext"
  cfg.tokenizer_name = "gpt2"
  cfg.min_sentences_per_document = 2
  cfg.min_sentence_tokens = 3

  # Not used by the TG branch, but kept so shared config plumbing works.
  cfg.ds_name = ""
  cfg.vocab_path = ""

  cfg.model = ml_collections.config_dict.create(
      # Geometry.
      D=768,
      H=12,
      N=12,
      F=None,  # None => 4*D for gelu, round_to_8(8/3*D) for swiglu
      # Sentence layout -> L = 1 + 64 + 1 = 66.
      max_sentence_tokens=64,
      sentence_tail_len=1,
      # "span" replaces sentence boundaries with fixed token windows and
      # overrides max_sentence_tokens/sentence_tail_len (EXPERIMENTS.md).
      segmentation="sentence",
      span_tokens=25,
      # Short-term memory.
      max_sentences_in_short_term=40,
      stm_cross_pos_mode="sinusoidal",
      stm_positional_weight=1.0,
      use_memory=True,  # False => memory-off ablation
      # 'external' = cross-attention; 'in_context' = memory as a token prefix.
      memory_mode="external",
      context_memory_slots=None,  # None => M when in_context, else 0
      detach_sreps_for_memory=False,  # True => break the recurrence (ablation)
      block_config=("S", "C") * 6,
      ffn_activation="swiglu",
      layer_norm_epsilon=1e-6,
      # No self gate: the reference freezes it at 1.0.
      memory_gate_init=1.0,
      # Dropout. `dropout` and `attn_dropout` are active from step 0;
      # token/memory/srep are scaled by the warm-in multiplier
      # (0.0 -> 0.5 -> 1.0 at dropout_half_step / dropout_full_step).
      dropout=0.1,  # models/tg/config.py (never an argparse flag)
      attn_dropout=0.2,  # args_tg.py --attn_dropout
      srep_dropout=0.15,  # args_tg.py --srep_dropout
      token_dropout=0.15,  # args_tg.py --token_dropout
      memory_dropout=0.0,  # args_tg.py --memory_dropout (inert at 0.0)
      dropout_scale=1.0,  # set per phase by the trainer
      dropout_half_step=2000,  # args_tg.py --dropout_half_step
      dropout_full_step=7000,  # args_tg.py --dropout_full_step
      # S_REP.
      srep_extraction_layer=6,
      srep_head_depth=1,
      srep_head_mult=1,
      srep_norm_target=1.0,
      srep_norm_margin=0.1,
      srep_norm_reg_weight=0.01,
      srep_pool_mode="eos",
      # BOS seeding.
      bos_replacement_mode="copy",
      bos_context_detach=False,
      # Special token ids; overwritten from the tokenizer at startup.
      bos_id=50258,
      eos_id=50259,
      eod_id=50260,
      pad_id=50257,
      # Initial stream cap; the curriculum overrides it per phase.
      max_sentences_per_stream=30,
      # Runtime.
      dtype="float32",
      fsdp_enabled=True,
      remat=False,
  )

  # Optimizer. Warmup is a RATIO of the schedule budget and starts from 0
  # (model_setup.py:build_scheduler), and the budget is recomputed whenever the
  # curriculum changes steps/epoch -- so there is no fixed num_train_steps.
  cfg.opt = ml_collections.config_dict.create(
      peak_learning_rate=2.5e-4,  # args_tg.py --lr
      warmup_ratio=0.004,  # args_tg.py --warmup_ratio
      cosine_min_lr_scale=0.1,  # args_tg.py --cosine_min_lr_scale
      decay_type="cosine",
      weight_decay=0.01,  # model_setup.py L357
      b1=0.9,
      b2=0.999,  # model_setup.py L359
      clip_by_global_norm=1.0,  # args_tg.py --grad_clip
      optimizer="adamw",
      num_train_steps=0,  # unused by the TG loop (epoch-driven); kept for logs
  )

  # Checkpointing. Checkpoints are written at every eval, and `max_to_keep`
  # retains the ones with the LOWEST eval_loss (not the most recent), so
  # `best_step()` gives the best-validation checkpoint.
  cfg.workdir = ""
  cfg.checkpoint = False  # launch scripts must opt in, or no model is saved
  cfg.checkpoint_restore_dir = None
  cfg.max_to_keep = 3

  # Eval runs once per epoch by construction (the curriculum changes
  # steps/epoch, so a fixed step cadence would drift relative to epochs).

  # Logging.
  # Validation cadence. The reference's adaptive schedule settles at 2 evals
  # per epoch; matching it removes a selection bias in best-val comparison.
  cfg.evals_per_epoch = 2
  cfg.max_steps_per_epoch = 0  # debug/probe cap; 0 = full epoch
  cfg.write_train_metrics_every_steps = 1
  cfg.write_perf_metrics_every_steps = 100
  cfg.write_to_xm_measurements = False
  cfg.log_internal_metrics = False

  return cfg
