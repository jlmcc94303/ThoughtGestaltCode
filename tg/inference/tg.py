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
"""TG inference: STM-threaded scoring and generation.

Port of `pipelines/inference/tg_inference.py`. Two entry points matter:

  * `score_next_token` -- run a document through the sentence loop and return
    the logits at one chosen position of one chosen sentence. This is the
    primitive the reversal-curse probe is built on
    (`tg_inference.py:score_next_token_logits`).
  * `generate` -- greedy/sampled continuation, appending each finished
    sentence's S_REP to the STM exactly as training does
    (`tg_inference.py:generate_with_stm`).

Both thread the STM in Python across sentence steps, so unlike the training
loop nothing is unrolled and `T` need not be known in advance.
"""

# pylint: disable=invalid-name,g-importing-member

import dataclasses
from typing import Any, Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from tg.data import tg as tg_data
from tg.models import tg_model
from tg.models.tg_config import TgConfig


# Tail written after the content tokens of a sentence.
#   'eos'     -- ordinary sentence: [EOS]
#   'eod_eos' -- final sentence of a document: [EOD] [EOS]
#   'open'    -- sentence still being generated: no tail yet
TAIL_EOS = 'eos'
TAIL_EOD_EOS = 'eod_eos'
TAIL_OPEN = 'open'
TAIL_MODES = (TAIL_EOS, TAIL_EOD_EOS, TAIL_OPEN)


@dataclasses.dataclass(frozen=True)
class SamplerConfig:
  """Decoding policy. Defaults are greedy, matching the reference rollout."""

  temperature: float = 0.0
  top_k: Optional[int] = 1
  top_p: Optional[float] = 1.0

  def is_greedy(self) -> bool:
    top_p_off = self.top_p is None or self.top_p >= 1.0
    return (self.temperature <= 0.0) or (self.top_k == 1 and top_p_off)


def check_specials_match_cfg(
    specials: tg_data.SpecialIds,
    cfg: TgConfig,
) -> None:
  """Fail loudly when the data-side ids disagree with the model-side ids.

  `build_sentence_row` writes markers from `specials`, but the model finds
  `[EOS]` with `cfg.eos_id` when deciding whether to commit an S_REP to memory.
  If the two disagree the model simply never sees an EOS: memory stays empty,
  every step scores as if there were no context, and a probe silently reports
  a no-memory model's numbers. That failure is invisible in the output, so it
  is checked rather than documented.
  """
  mismatches = [
      (name, data_id, cfg_id)
      for name, data_id, cfg_id in (
          ('bos', specials.bos, cfg.bos_id),
          ('eos', specials.eos, cfg.eos_id),
          ('eod', specials.eod, cfg.eod_id),
          ('pad', specials.pad, cfg.pad_id),
      )
      if int(data_id) != int(cfg_id)
  ]
  if mismatches:
    detail = ', '.join(
        f'{n}: specials={d} but cfg={c}' for n, d, c in mismatches
    )
    raise ValueError(
        'SpecialIds disagree with TgConfig token ids -- the model would never '
        f'detect [EOS] and memory would stay empty ({detail})'
    )


def build_sentence_row(
    content_ids: Sequence[int],
    specials: tg_data.SpecialIds,
    cfg: TgConfig,
    tail_mode: str = TAIL_EOS,
) -> tuple[np.ndarray, np.ndarray, int]:
  """`[BOS] + content + tail`, padded to `cfg.L`.

  Returns `(ids_L, mask_L, prefix_len)` where `prefix_len` is the number of
  real tokens *before* the tail -- i.e. the position whose logits predict the
  next content token. Layout is identical to `tg_data.tensorize_document` so a
  probe scores the model on exactly the rows it was trained on.
  """
  if tail_mode not in TAIL_MODES:
    raise ValueError(f'tail_mode must be one of {TAIL_MODES}; got {tail_mode!r}')
  content = list(content_ids)[: cfg.max_sentence_tokens]
  body = [specials.bos] + content
  if tail_mode == TAIL_EOS:
    tail = [specials.eos]
  elif tail_mode == TAIL_EOD_EOS:
    tail = [specials.eod, specials.eos]
  else:
    tail = []
  prefix_len = len(body)
  full = body + tail
  if len(full) > cfg.L:
    raise ValueError(
        f'sentence of {len(full)} tokens exceeds L={cfg.L}; '
        f'max_sentence_tokens={cfg.max_sentence_tokens}'
    )
  ids = np.full((cfg.L,), specials.pad, dtype=np.int32)
  mask = np.zeros((cfg.L,), dtype=np.int32)
  ids[: len(full)] = full
  mask[: len(full)] = 1
  return ids, mask, prefix_len


def encode_sentences(
    texts: Sequence[str],
    tokenizer: Any,
    specials: tg_data.SpecialIds,
    cfg: TgConfig,
    final_is_document_end: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
  """Text sentences -> `[S, L]` id/mask rows ready for the sentence loop."""
  rows_ids, rows_mask = [], []
  for i, text in enumerate(texts):
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    last = i == len(texts) - 1
    tail = TAIL_EOD_EOS if (last and final_is_document_end) else TAIL_EOS
    row_ids, row_mask, _ = build_sentence_row(ids, specials, cfg, tail)
    rows_ids.append(row_ids)
    rows_mask.append(row_mask)
  if not rows_ids:
    return (
        np.zeros((0, cfg.L), np.int32),
        np.zeros((0, cfg.L), np.int32),
    )
  return np.stack(rows_ids), np.stack(rows_mask)


class TgRunner:
  """Threads the STM across sentence steps for a single document (B = 1).

  The training loop unrolls `T` static steps into one graph; for inference we
  instead keep `mem` in Python and call a jitted single-sentence step, so
  documents of any length cost one compile.
  """

  def __init__(
      self,
      apply_fn: Callable[..., tg_model.TgStepOutput],
      params: Any,
      cfg: TgConfig,
      batch_size: int = 1,
  ):
    self.cfg = cfg
    self.params = params
    self.B = int(batch_size)
    self._apply = jax.jit(
        lambda v, ids, mask, kv, valid, ctx, ctx_ok, steps: apply_fn(
            v, ids, mask, kv, valid, ctx, ctx_ok, True, steps
        )
    )
    self.reset()

  def reset(self) -> None:
    """Fresh STM -- the document boundary reset the recurrence depends on."""
    self.mem = tg_model.init_memory(self.B, self.cfg)
    self.bos_ctx = jnp.zeros((self.B, self.cfg.D), dtype=self.cfg.dtype)
    self.bos_ctx_valid = jnp.zeros((self.B,), dtype=jnp.bool_)
    # Sentence index, needed by multi_eos_pos_mode='grouped' to give the K
    # vectors of one sentence a shared cross-attention position. Named
    # `step_index` because `step` is this class's forward method.
    self.step_index = 0

  def step(
      self,
      ids_BxL: np.ndarray,
      mask_BxL: np.ndarray,
      write_memory: bool = True,
  ) -> tg_model.TgStepOutput:
    """Run one sentence; optionally commit its S_REP to the STM.

    `write_memory=False` scores a sentence without letting it influence later
    ones -- what a probe wants for the query sentence.
    """
    out = self._apply(
        {'params': self.params},
        jnp.asarray(ids_BxL),
        jnp.asarray(mask_BxL),
        self.mem.kv_BxMxD,
        self.mem.valid_BxM,
        self.bos_ctx,
        self.bos_ctx_valid,
        self.mem.step_BxM,
    )
    if write_memory:
      write_B = out.has_eos_B
      if self.cfg.use_memory:
        self.mem = tg_model.push_memory(
            self.mem, out.srep_BxD, write_B, step=self.step_index)
      # Only committed sentences advance the index, so a scored-but-not-stored
      # query does not consume a position.
      self.step_index += 1
      if self.cfg.bos_replacement_mode == 'copy':
        self.bos_ctx = out.srep_BxD
        self.bos_ctx_valid = write_B
    return out


def score_next_token(
    apply_fn: Callable[..., tg_model.TgStepOutput],
    params: Any,
    cfg: TgConfig,
    context_rows: tuple[np.ndarray, np.ndarray],
    query_content_ids: Sequence[int],
    specials: tg_data.SpecialIds,
) -> np.ndarray:
  """Logits at the first answer position of a query sentence.

  `context_rows` are `[S, L]` id/mask rows that are run first and written to
  memory; `query_content_ids` is the unfinished query, scored *without* being
  written back. The returned vector is the distribution over the token that
  would come next -- `logits[prefix_len - 1]`, since position i predicts i + 1.
  """
  check_specials_match_cfg(specials, cfg)
  runner = TgRunner(apply_fn, params, cfg, batch_size=1)
  ctx_ids, ctx_mask = context_rows
  for s in range(ctx_ids.shape[0]):
    runner.step(ctx_ids[s : s + 1], ctx_mask[s : s + 1], write_memory=True)

  q_ids, q_mask, prefix_len = build_sentence_row(
      query_content_ids, specials, cfg, TAIL_OPEN
  )
  out = runner.step(q_ids[None, :], q_mask[None, :], write_memory=False)
  return np.asarray(out.logits_BxLxV[0, prefix_len - 1])


def _apply_top_k(logits: np.ndarray, top_k: Optional[int]) -> np.ndarray:
  if not top_k or top_k <= 0 or top_k >= logits.shape[-1]:
    return logits
  cutoff = np.partition(logits, -top_k)[-top_k]
  return np.where(logits < cutoff, -np.inf, logits)


def _apply_top_p(logits: np.ndarray, top_p: Optional[float]) -> np.ndarray:
  if top_p is None or top_p >= 1.0 or top_p <= 0.0:
    return logits
  order = np.argsort(logits)[::-1]
  probs = np.exp(logits[order] - logits[order].max())
  probs /= probs.sum()
  keep = np.cumsum(probs) <= top_p
  keep[0] = True  # always keep the argmax
  blocked = order[~keep]
  out = logits.copy()
  out[blocked] = -np.inf
  return out


def sample_next_token(
    logits: np.ndarray,
    sampler: SamplerConfig,
    rng: Optional[np.random.Generator] = None,
) -> int:
  """`tg_inference.py:sample_next_token`."""
  if sampler.is_greedy():
    return int(np.argmax(logits))
  scaled = logits / max(1e-6, sampler.temperature)
  scaled = _apply_top_p(_apply_top_k(scaled, sampler.top_k), sampler.top_p)
  probs = np.exp(scaled - np.nanmax(scaled))
  probs = np.where(np.isfinite(probs), probs, 0.0)
  total = probs.sum()
  if total <= 0:
    return int(np.argmax(logits))
  probs /= total
  rng = rng or np.random.default_rng()
  return int(rng.choice(probs.shape[-1], p=probs))


@dataclasses.dataclass
class GeneratedSentence:
  token_ids: list[int]
  text: str
  hit_eos: bool
  hit_eod: bool


def generate(
    apply_fn: Callable[..., tg_model.TgStepOutput],
    params: Any,
    cfg: TgConfig,
    specials: tg_data.SpecialIds,
    tokenizer: Any,
    context_rows: Optional[tuple[np.ndarray, np.ndarray]] = None,
    max_sentences: int = 10,
    max_tokens: int = 200,
    sampler: Optional[SamplerConfig] = None,
    seed: int = 0,
) -> list[GeneratedSentence]:
  """Generate sentence by sentence, committing each S_REP to the STM.

  Mirrors `generate_with_stm`: within a sentence the model is re-run on the
  growing prefix (no KV cache -- the reference has none either), and at `[EOS]`
  the finished row is re-run with its real tail so the S_REP written to memory
  is the one training would have written.
  """
  check_specials_match_cfg(specials, cfg)
  sampler = sampler or SamplerConfig()
  rng = np.random.default_rng(seed)
  runner = TgRunner(apply_fn, params, cfg, batch_size=1)

  if context_rows is not None:
    ctx_ids, ctx_mask = context_rows
    for s in range(ctx_ids.shape[0]):
      runner.step(ctx_ids[s : s + 1], ctx_mask[s : s + 1], write_memory=True)

  produced: list[GeneratedSentence] = []
  budget = int(max_tokens)
  for _ in range(int(max_sentences)):
    if budget <= 0:
      break
    content: list[int] = []
    hit_eos = hit_eod = False
    while len(content) < cfg.max_sentence_tokens and budget > 0:
      ids, mask, prefix_len = build_sentence_row(
          content, specials, cfg, TAIL_OPEN
      )
      out = runner.step(ids[None, :], mask[None, :], write_memory=False)
      logits = np.asarray(out.logits_BxLxV[0, prefix_len - 1])
      nxt = sample_next_token(logits, sampler, rng)
      budget -= 1
      if nxt == specials.eos:
        hit_eos = True
        break
      if nxt == specials.eod:
        hit_eod = True
        continue
      if nxt == specials.pad or nxt == specials.bos:
        continue
      content.append(nxt)

    # Re-run the finished sentence with its real tail so the S_REP committed to
    # memory matches what training would have produced for this sentence.
    tail = TAIL_EOD_EOS if hit_eod else TAIL_EOS
    ids, mask, _ = build_sentence_row(content, specials, cfg, tail)
    runner.step(ids[None, :], mask[None, :], write_memory=True)

    produced.append(
        GeneratedSentence(
            token_ids=content,
            text=tokenizer.decode(content) if tokenizer is not None else '',
            hit_eos=hit_eos,
            hit_eod=hit_eod,
        )
    )
    if hit_eod:
      break
  return produced
