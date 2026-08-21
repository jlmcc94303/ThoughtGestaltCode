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
"""Text -> documents -> sentence tensors for TG.

The tensor layout is a direct port of `pipelines/data/transform.py`
(`process_documents_for_training` + `_enforce_last_sentence_terminals`):

  * every sentence is `[BOS] + <= max_sentence_tokens content + tail`, padded
    with `[PAD]` to `1 + max_sentence_tokens + sentence_tail_len` (= 66);
  * the tail is `[EOS]` (plus `[PAD]` filler when `sentence_tail_len > 1`);
  * the *final* sentence of a document instead ends `... [EOD] [EOS]`
    (`--doc_terminal_order eod_eos`), truncating content from the left --
    keeping `[BOS]` -- to make room;
  * `attention_mask` is 1 on real tokens and 0 on padding.

Sentence splitting here is a naive `.!?` regex; a SaT model splitter can be
run offline instead and its output supplied via the corpus cache.
"""

# pylint: disable=invalid-name,g-importing-member

import dataclasses
import json
import math
import random
import re
from typing import Any, Iterator, Optional, Sequence

import numpy as np
from tg.models import tg_config


# `core/tokenizer_setup.py`
SPECIAL_TOKENS = {
    'pad_token': '[PAD]',
    'bos_token': '[BOS]',
    'eos_token': '[EOS]',
}
EOD_TOKEN = '[EOD]'


def build_eos_tokens(count: int) -> list[str]:
  """`[EOS]`, `[EOS2]`, ... `[EOSK]` -- `core/multi_eos.py:build_eos_tokens`."""
  count = max(1, int(count))
  return ['[EOS]'] + [f'[EOS{i}]' for i in range(2, count + 1)]


def ensure_multi_eos_tokens(tokenizer: Any, count: int) -> Any:
  """Idempotently add `[EOS2]..[EOSK]` as additional special tokens."""
  count = max(1, int(count))
  if count <= 1:
    return tokenizer
  extra = build_eos_tokens(count)[1:]
  existing = list(
      getattr(tokenizer, 'additional_special_tokens', None) or [])
  added = [t for t in extra if t not in existing]
  if added:
    tokenizer.add_special_tokens(
        {'additional_special_tokens': existing + added})
  return tokenizer


def multi_eos_ids(tokenizer: Any, count: int) -> list[int]:
  """Ids of `[EOS]`, `[EOS2]`, ... in order; raises if any is missing."""
  out = []
  for tok in build_eos_tokens(count):
    tid = tokenizer.convert_tokens_to_ids(tok)
    if tid is None or tid == tokenizer.unk_token_id:
      raise ValueError(
          f"tokenizer is missing {tok!r}; call ensure_multi_eos_tokens first")
    out.append(int(tid))
  return out

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
_PARAGRAPH_SPLIT_RE = re.compile(r'\n\s*\n')

# Document segmentation modes, mirroring the reference's --document_format.
WIKITEXT = 'wikitext'
PARAGRAPH = 'paragraph'
DOCUMENT_FORMATS = (WIKITEXT, PARAGRAPH)

# Segmentation modes: how a document is cut into the units TG recurs over.
# 'sentence' is the paper's model; 'span' is the fixed token-span ablation of
# paper section 3.3 / Table 4 (`args_tg.py --pseudo_sentence_mode`), which
# replaces sentence boundaries with arbitrary N-token windows so that memory
# entries summarize token blocks rather than semantically coherent units.
SEGMENT_SENTENCE = 'sentence'
SEGMENT_SPAN = 'span'
SEGMENTATIONS = (SEGMENT_SENTENCE, SEGMENT_SPAN)


@dataclasses.dataclass(frozen=True)
class SpecialIds:
  bos: int
  eos: int
  eod: int
  pad: int

  @classmethod
  def defaults(cls) -> 'SpecialIds':
    return cls(
        bos=tg_config.DEFAULT_BOS_ID,
        eos=tg_config.DEFAULT_EOS_ID,
        eod=tg_config.DEFAULT_EOD_ID,
        pad=tg_config.DEFAULT_PAD_ID,
    )

  @classmethod
  def from_config(cls, cfg: tg_config.TgConfig) -> 'SpecialIds':
    return cls(bos=cfg.bos_id, eos=cfg.eos_id, eod=cfg.eod_id, pad=cfg.pad_id)


@dataclasses.dataclass
class Stream:
  """One training stream: a contiguous run of sentences sharing an STM."""

  ids_SxL: np.ndarray
  mask_SxL: np.ndarray
  doc_id: str = ''

  @property
  def num_sentences(self) -> int:
    return int(self.ids_SxL.shape[0])


# --------------------------------------------------------------------------- #
# Tokenizer (mirrors core/tokenizer_setup.py)
# --------------------------------------------------------------------------- #
def build_tokenizer(model_name: str = 'gpt2') -> Any:
  """GPT2TokenizerFast + the four TG specials. Requires `transformers`."""
  from transformers import GPT2TokenizerFast  # pylint: disable=g-import-not-at-top

  tok = GPT2TokenizerFast.from_pretrained(model_name)
  new_tokens = dict(SPECIAL_TOKENS)
  additional = list(getattr(tok, 'additional_special_tokens', None) or [])
  if EOD_TOKEN not in additional:
    additional.append(EOD_TOKEN)
  new_tokens['additional_special_tokens'] = additional
  tok.add_special_tokens(new_tokens)
  if tok.pad_token is None and tok.eos_token is not None:
    tok.pad_token = tok.eos_token
  return tok


def special_ids_from_tokenizer(tokenizer: Any) -> SpecialIds:
  """`core/tokenizer_setup.py:get_special_token_ids`."""
  eos = int(tokenizer.eos_token_id)
  bos = int(tokenizer.bos_token_id if tokenizer.bos_token_id is not None else eos)
  pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos)
  eod = int(tokenizer.convert_tokens_to_ids(EOD_TOKEN))
  if eod is None or eod < 0:
    eod = eos
  return SpecialIds(bos=bos, eos=eos, eod=eod, pad=pad)


# --------------------------------------------------------------------------- #
# Text -> documents -> sentences
# --------------------------------------------------------------------------- #
def _is_top_title(line: str) -> bool:
  """True iff `line` is a top-level WikiText article title `= Title =`.

  Port of `pipelines/data/io.py:_is_top_title`. Rejects section headers
  (`== ... ==`, `=== ... ===`) and their spaced variants (`= = ... = =`), so a
  document is a whole article rather than a section.
  """
  if not line:
    return False
  t = str(line).strip()
  if not (t.startswith('=') and t.endswith('=')):
    return False
  t_nospace = t.replace(' ', '')
  if t_nospace.startswith('==') or t_nospace.endswith('=='):
    return False
  if t_nospace.count('=') != 2:
    return False
  return len(t.strip('=').strip()) > 0


def split_wikitext_articles(raw: str) -> list[str]:
  """Split a WikiText dump into articles at `= Title =` lines.

  Port of `pipelines/data/io.py:extract_documents_from_text_by_title_rows`:
  the title line is kept as the first line of its article, subsections stay
  inside, and any content before the first top-level title is dropped.
  """
  out: list[str] = []
  current: list[str] = []
  have_title = False

  def flush():
    if have_title and current:
      text = '\n'.join(current)
      if text.strip():
        out.append(text)

  for raw_line in raw.splitlines():
    line = '' if raw_line is None else str(raw_line)
    if _is_top_title(line):
      flush()
      current = [line.rstrip()]
      have_title = True
    elif have_title:
      current.append(line.rstrip())
  flush()
  return out


def read_documents(path: str, document_format: str = WIKITEXT) -> list[str]:
  """Read a corpus file into documents.

  `.jsonl` always means one record (its `text` field) per document. Otherwise
  `document_format` decides:

    * `"wikitext"` -- split at `= Title =` lines, i.e. one document per
      article. This is what the PyTorch pipeline does (`--document_format
      wikitext`) and what TG's recurrence assumes: the short-term memory is
      reset per document, so document boundaries must be article boundaries.
    * `"paragraph"` -- split on blank lines. Only appropriate for corpora that
      really are one document per paragraph.
  """
  if document_format not in DOCUMENT_FORMATS:
    raise ValueError(
        f'document_format must be one of {DOCUMENT_FORMATS}; '
        f'got {document_format!r}'
    )
  with open(path, 'r', encoding='utf-8') as f:
    raw = f.read()

  if path.endswith('.jsonl'):
    docs = []
    for line in raw.splitlines():
      line = line.strip()
      if not line:
        continue
      obj = json.loads(line)
      text = obj.get('text', '') if isinstance(obj, dict) else str(obj)
      if text.strip():
        docs.append(text)
    return docs

  if document_format == WIKITEXT:
    docs = split_wikitext_articles(raw)
    if not docs:
      raise ValueError(
          f'{path}: document_format="wikitext" found no "= Title =" lines, so '
          'the file would yield zero documents. Pass '
          'document_format="paragraph" if this corpus is not WikiText.'
      )
    return docs

  return [d.strip() for d in _PARAGRAPH_SPLIT_RE.split(raw) if d.strip()]


def split_sentences(text: str) -> list[str]:
  """Naive `.!?` sentence splitter (v1 stand-in for the SaT splitter)."""
  text = ' '.join(text.split())
  if not text:
    return []
  return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def encode_sentences(
    sentences: Sequence[str],
    tokenizer: Any,
    max_sentence_tokens: int = 64,
    min_sentence_tokens: Optional[int] = None,
) -> list[list[int]]:
  """Tokenize sentences, truncating to `max_sentence_tokens` content tokens.

  The reference re-splits over-long sentences into fragments; v1 truncates.
  """
  out = []
  for s in sentences:
    ids = list(tokenizer.encode(s, add_special_tokens=False))
    if not ids:
      continue
    if min_sentence_tokens is not None and len(ids) < int(min_sentence_tokens):
      continue
    out.append(ids[:max_sentence_tokens])
  return out


# --------------------------------------------------------------------------- #
# Fixed token-span segmentation (paper 3.3 ablation)
# --------------------------------------------------------------------------- #
def spans_from_token_stream(
    token_ids: Sequence[int],
    span_tokens: int,
) -> list[list[int]]:
  """Chop a flat document token stream into consecutive `span_tokens` blocks.

  This is the ablation unit: each block is later wrapped in `[BOS] ... [EOS]`
  and compressed into one memory vector exactly like a sentence, so the *only*
  thing that changes versus TG is that memory entries now summarize arbitrary
  token windows. The reference reaches this via `--pseudo_sentence_mode`
  (`args_tg.py` L478-482), which also pins `sentence_tail_len = 1`.

  The trailing partial block is kept -- dropping it would silently discard up
  to `span_tokens - 1` tokens per document and make the token budget differ
  from the sentence-mode run it is being compared against.
  """
  if span_tokens <= 0:
    raise ValueError(f'span_tokens must be > 0; got {span_tokens}')
  ids = list(token_ids)
  return [ids[i:i + span_tokens] for i in range(0, len(ids), span_tokens)]


def segment_into_spans(
    text: str,
    tokenizer: Any,
    span_tokens: int,
) -> list[list[int]]:
  """Encode a document as one stream, then cut it into fixed token spans."""
  ids = list(tokenizer.encode(text, add_special_tokens=False))
  if not ids:
    return []
  return spans_from_token_stream(ids, span_tokens)


def respan_cached_document(
    sentences: Sequence[Sequence[int]],
    span_tokens: int,
) -> list[list[int]]:
  """Re-cut an already-sentence-split cached document into fixed spans.

  The SaT cache stores per-sentence token ids; concatenating them recovers the
  document token stream, so the span ablation can reuse the same cache as the
  sentence runs rather than re-tokenizing the corpus. Token counts therefore
  match the sentence-mode run exactly, which is what makes the comparison fair.
  """
  flat: list[int] = []
  for sent in sentences:
    flat.extend(sent)
  return spans_from_token_stream(flat, span_tokens)


# --------------------------------------------------------------------------- #
# Tensorization (transform.py)
# --------------------------------------------------------------------------- #
def _keep_tail_with_bos(tokens: list[int], max_len: int, bos_id: int) -> list[int]:
  """`transform.py:_keep_tail_with_bos`."""
  if max_len <= 0:
    return [bos_id]
  if len(tokens) <= max_len:
    return tokens
  if tokens and tokens[0] == bos_id:
    if max_len == 1:
      return [bos_id]
    return [bos_id] + tokens[-(max_len - 1):]
  return tokens[-max_len:]


def _apply_document_terminal_tokens(
    valid_tokens: list[int],
    specials: SpecialIds,
    target_len: int,
    doc_terminal_order: str = 'eod_eos',
) -> tuple[list[int], list[int]]:
  """`transform.py:_apply_document_terminal_tokens` (single-EOS case)."""
  tokens = list(valid_tokens) if valid_tokens else [specials.bos]
  while tokens and tokens[-1] in (specials.eos, specials.eod):
    tokens.pop()

  parts = [p for p in str(doc_terminal_order or '').split('_') if p]
  if not parts:
    parts = ['eod', 'eos']
  terminal_ids = []
  for part in parts:
    if part == 'eos':
      terminal_ids.append(specials.eos)
    elif part == 'eod':
      terminal_ids.append(specials.eod)
    else:
      raise ValueError(f'Unsupported terminal token {part!r}')

  if target_len < 1 + len(terminal_ids):
    raise ValueError(
        f'Context length {target_len} is too small for BOS + '
        f'{len(terminal_ids)} terminal tokens.'
    )
  max_base_len = max(1, target_len - len(terminal_ids))
  tokens = _keep_tail_with_bos(tokens, max_base_len, specials.bos)
  final_tokens = tokens + terminal_ids
  if len(final_tokens) > target_len:
    final_tokens = _keep_tail_with_bos(final_tokens, target_len, specials.bos)

  pad_len = target_len - len(final_tokens)
  ids = final_tokens + [specials.pad] * pad_len
  mask = [1] * len(final_tokens) + [0] * pad_len
  return ids, mask


def tensorize_document(
    sentence_token_ids: Sequence[Sequence[int]],
    specials: SpecialIds,
    *,
    max_sentence_tokens: int = 64,
    sentence_tail_len: int = 1,
    doc_terminal_order: str = 'eod_eos',
    apply_document_terminals: bool = True,
    eos_ids: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Build `[S, L]` id/mask tensors for one document.

  `eos_ids` defaults to `[specials.eos]`; pass the full multi-EOS list to write
  `[EOS] [EOS2] ...` into the tail.
  """
  if sentence_tail_len < 1:
    raise ValueError('sentence_tail_len must be >= 1')
  # Multi-EOS writes every EOS id into the tail; the remaining slots (the
  # reserved [EOD] position) stay [PAD] and unmasked.
  eos_tail = list(eos_ids) if eos_ids else [specials.eos]
  if len(eos_tail) > sentence_tail_len:
    raise ValueError(
        f'{len(eos_tail)} EOS ids do not fit in sentence_tail_len='
        f'{sentence_tail_len}')
  n_pad = sentence_tail_len - len(eos_tail)
  tail_ids = eos_tail + [specials.pad] * n_pad
  tail_mask = [1] * len(eos_tail) + [0] * n_pad
  target_len = 1 + max_sentence_tokens + sentence_tail_len

  ids_rows, mask_rows = [], []
  for tok_ids in sentence_token_ids:
    content = [specials.bos] + list(tok_ids)[:max_sentence_tokens]
    full = content + tail_ids
    mask = [1] * len(content) + tail_mask
    pad_len = target_len - len(full)
    if pad_len > 0:
      full = full + [specials.pad] * pad_len
      mask = mask + [0] * pad_len
    ids_rows.append(full)
    mask_rows.append(mask)

  if ids_rows and apply_document_terminals:
    valid_len = int(np.sum(mask_rows[-1]))
    valid_tokens = ids_rows[-1][:valid_len] if valid_len > 0 else [specials.bos]
    ids_rows[-1], mask_rows[-1] = _apply_document_terminal_tokens(
        valid_tokens, specials, target_len, doc_terminal_order
    )

  if not ids_rows:
    return (
        np.zeros((0, target_len), dtype=np.int32),
        np.zeros((0, target_len), dtype=np.int32),
    )
  return (
      np.asarray(ids_rows, dtype=np.int32),
      np.asarray(mask_rows, dtype=np.int32),
  )


def chunk_into_streams(
    ids_SxL: np.ndarray,
    mask_SxL: np.ndarray,
    max_sentences_per_stream: int = 30,
    doc_id: str = '',
) -> list[Stream]:
  """Split a document into consecutive streams of <= N sentences, no overlap.

  Mirrors `pipelines/data/chunking_common.py:chunk_processed_documents`: the
  document terminals live on the true last sentence, so only the final chunk
  carries `[EOD] [EOS]`. Each stream starts with a fresh STM.
  """
  S = int(ids_SxL.shape[0])
  if S == 0:
    return []
  if S <= max_sentences_per_stream:
    return [Stream(ids_SxL, mask_SxL, doc_id)]
  streams = []
  for j, start in enumerate(range(0, S, max_sentences_per_stream)):
    end = min(start + max_sentences_per_stream, S)
    streams.append(
        Stream(ids_SxL[start:end], mask_SxL[start:end], f'{doc_id}_chunk_{j}')
    )
  return streams


def load_cached_corpus(path: str) -> tuple[list[list[list[int]]], dict]:
  """Load a corpus preprocessed by `preprocess_corpus.py`.

  Returns `(documents, meta)` where each document is a list of sentences and
  each sentence is a list of token ids. The SaT splitter (`sat-3l-sm`,
  threshold 0.15, `short_sentence_mode=merge`) runs once offline under the
  PyTorch env; this just replays its output.
  """
  with np.load(path, allow_pickle=False) as z:
    tokens = z['tokens']
    sent_offsets = z['sent_offsets']
    doc_sent_offsets = z['doc_sent_offsets']
    meta = json.loads(str(z['meta']))
  docs: list[list[list[int]]] = []
  for d in range(len(doc_sent_offsets) - 1):
    s0, s1 = int(doc_sent_offsets[d]), int(doc_sent_offsets[d + 1])
    docs.append([
        tokens[int(sent_offsets[s]):int(sent_offsets[s + 1])].tolist()
        for s in range(s0, s1)
    ])
  return docs, meta


def streams_from_cached_corpus(
    docs: Sequence[Sequence[Sequence[int]]],
    specials: SpecialIds,
    cfg: tg_config.TgConfig,
    chunk_size: Optional[int] = None,
    min_sentences_per_document: int = 2,
) -> list[Stream]:
  """Tensorize + chunk a cached corpus at a given chunk size.

  `chunk_size` is a parameter rather than a config constant because the chunk
  curriculum changes it at epoch boundaries; callers re-run this per phase.
  """
  if chunk_size is None:
    chunk_size = cfg.max_sentences_per_stream
  streams: list[Stream] = []
  for i, sentences in enumerate(docs):
    if len(sentences) < min_sentences_per_document:
      continue
    if cfg.segmentation == SEGMENT_SPAN:
      sentences = respan_cached_document(sentences, cfg.span_tokens)
    ids_SxL, mask_SxL = tensorize_document(
        sentences,
        specials,
        max_sentence_tokens=cfg.max_sentence_tokens,
        sentence_tail_len=cfg.sentence_tail_len,
        eos_ids=cfg.multi_eos_ids,
    )
    streams.extend(
        chunk_into_streams(ids_SxL, mask_SxL, int(chunk_size), f'doc_{i}')
    )
  return streams


def streams_from_text(
    text: str,
    tokenizer: Any,
    specials: SpecialIds,
    cfg: tg_config.TgConfig,
    doc_id: str = '',
    min_sentence_tokens: Optional[int] = None,
) -> list[Stream]:
  """Full pipeline for one document: split -> encode -> tensorize -> chunk.

  Under `cfg.segmentation == 'span'` the sentence splitter is bypassed
  entirely and the document is cut into fixed `cfg.span_tokens` blocks
  (paper section 3.3 ablation).
  """
  if cfg.segmentation == SEGMENT_SPAN:
    token_ids = segment_into_spans(text, tokenizer, cfg.span_tokens)
  else:
    sentences = split_sentences(text)
    token_ids = encode_sentences(
        sentences,
        tokenizer,
        max_sentence_tokens=cfg.max_sentence_tokens,
        min_sentence_tokens=min_sentence_tokens,
    )
  ids_SxL, mask_SxL = tensorize_document(
      token_ids,
      specials,
      max_sentence_tokens=cfg.max_sentence_tokens,
      sentence_tail_len=cfg.sentence_tail_len,
      eos_ids=cfg.multi_eos_ids,
  )
  return chunk_into_streams(
      ids_SxL, mask_SxL, cfg.max_sentences_per_stream, doc_id
  )


def streams_from_file(
    path: str,
    tokenizer: Any,
    specials: SpecialIds,
    cfg: tg_config.TgConfig,
    min_sentences_per_document: int = 2,
    min_sentence_tokens: Optional[int] = None,
    document_format: str = WIKITEXT,
) -> list[Stream]:
  """Read a corpus file and return every training stream in it."""
  streams: list[Stream] = []
  for i, doc in enumerate(read_documents(path, document_format)):
    doc_streams = streams_from_text(
        doc, tokenizer, specials, cfg, f'doc_{i}', min_sentence_tokens
    )
    total = sum(s.num_sentences for s in doc_streams)
    if total < min_sentences_per_document:
      continue
    streams.extend(doc_streams)
  return streams


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #
def make_batch(
    streams: Sequence[Stream],
    pad_to_sentences: Optional[int] = None,
    pad_to_batch: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Pad a list of streams to a common sentence count (and batch size).

  Returns `(sentences_BxTxL, masks_BxTxL, lengths_B)`. Padded steps and padded
  rows are all zeros with an all-zero mask and `lengths == 0`, so they
  contribute no loss and never touch the STM (see `run_sentence_loop`).

  `pad_to_sentences` fixes `T` and `pad_to_batch` fixes `B`. Token-budget
  batching produces variable `B`, so pinning both is what keeps jit from
  recompiling the (large, unrolled) sentence loop for every new shape.
  """
  if not streams:
    raise ValueError('make_batch requires at least one stream')
  B = len(streams)
  T = max(s.num_sentences for s in streams)
  if pad_to_sentences is not None:
    if T > pad_to_sentences:
      raise ValueError(
          f'stream has {T} sentences, more than pad_to_sentences='
          f'{pad_to_sentences}'
      )
    T = int(pad_to_sentences)
  if pad_to_batch is not None:
    if B > pad_to_batch:
      raise ValueError(
          f'batch has {B} streams, more than pad_to_batch={pad_to_batch}'
      )
    B = int(pad_to_batch)
  L = int(streams[0].ids_SxL.shape[1])

  sentences = np.zeros((B, T, L), dtype=np.int32)
  masks = np.zeros((B, T, L), dtype=np.int32)
  lengths = np.zeros((B,), dtype=np.int32)
  for b, s in enumerate(streams):
    n = s.num_sentences
    sentences[b, :n] = s.ids_SxL
    masks[b, :n] = s.mask_SxL
    lengths[b] = n
  return sentences, masks, lengths


def stream_token_counts(streams: Sequence[Stream]) -> list[int]:
  """`loss_tokens_with_special` per stream: its non-padding token count.

  This is the statistic the reference buckets and budgets on
  (`--bucket_stat_key` / `--mixed_bucket_budget_key`, both defaulting to
  `loss_tokens_with_special`), computed per chunk after chunking.
  """
  return [int(s.mask_SxL.sum()) for s in streams]


class LengthBucketBatchSampler:
  """Token-budget batch sampler.

  Port of `core/data/data_sampler.py:LengthBucketBatchSampler` in its uniform
  (token-budget) mode, which is what `--bucket_sampler` selects. Instead of a
  fixed number of documents per batch, each batch accumulates examples until it
  hits `target_budget_per_batch` tokens or `max_examples_per_batch` examples.

  Examples are shuffled, bucketed by length (`bucket_width`), then walked
  longest-bucket-first and dealt round-robin into a pre-allocated pool of
  `estimated_batches` batches, so batches end up with similar token loads and
  similar-length documents sit together.

  Yields lists of stream indices; batch sizes vary.
  """

  def __init__(
      self,
      lengths: Sequence[int],
      batch_size: int,
      bucket_width: int,
      seed: int = 42,
      budget_lengths: Optional[Sequence[int]] = None,
      target_budget_per_batch: Optional[int] = None,
      estimated_batches: Optional[int] = None,
      max_examples_per_batch: Optional[int] = None,
  ):
    if batch_size <= 0:
      raise ValueError('batch_size must be > 0 for LengthBucketBatchSampler.')
    if bucket_width <= 0:
      raise ValueError('bucket_width must be > 0 for LengthBucketBatchSampler.')
    if target_budget_per_batch is None or int(target_budget_per_batch) <= 0:
      raise ValueError(
          'target_budget_per_batch must be a positive int for uniform bucket '
          'sampling.'
      )
    self.lengths = [int(max(0, x)) for x in lengths]
    self.batch_size = int(batch_size)
    self.bucket_width = int(bucket_width)
    self.seed = int(seed)
    self.epoch = 0
    if budget_lengths is None:
      self.budget_lengths = list(self.lengths)
    else:
      if len(budget_lengths) != len(self.lengths):
        raise ValueError('budget_lengths must match the number of examples.')
      self.budget_lengths = [int(max(0, x)) for x in budget_lengths]
    self.target_budget_per_batch = int(target_budget_per_batch)
    self.dataset_size = len(self.lengths)
    self.total_budget_sum = sum(self.budget_lengths)
    self.avg_budget_per_example = (
        float(self.total_budget_sum) / float(self.dataset_size)
        if self.dataset_size and self.total_budget_sum > 0
        else 0.0
    )

    # `data_setup.py` derives both of these before constructing the sampler.
    if estimated_batches and int(estimated_batches) > 0:
      self.estimated_batches = int(estimated_batches)
    elif self.total_budget_sum > 0:
      self.estimated_batches = max(
          1, math.ceil(self.total_budget_sum / self.target_budget_per_batch)
      )
    else:
      self.estimated_batches = max(
          1, math.ceil(self.dataset_size / max(1, self.batch_size))
      )
    if max_examples_per_batch and int(max_examples_per_batch) > 0:
      self.max_examples_per_batch = int(max_examples_per_batch)
    else:
      avg = self.avg_budget_per_example
      if avg <= 0:
        avg = float(self.target_budget_per_batch) / max(1, self.batch_size)
      self.max_examples_per_batch = max(
          1, math.ceil(self.target_budget_per_batch / max(1.0, avg))
      )

  def __len__(self) -> int:
    if self.total_budget_sum <= 0:
      return math.ceil(self.dataset_size / max(1, self.batch_size))
    return max(
        1, math.ceil(self.total_budget_sum / self.target_budget_per_batch)
    )

  def set_epoch(self, epoch: int) -> None:
    self.epoch = int(max(0, epoch))

  def __iter__(self) -> Iterator[list[int]]:
    if self.dataset_size == 0:
      return iter(())
    rng = random.Random(self.seed + self.epoch)
    indices = list(range(self.dataset_size))
    rng.shuffle(indices)
    bucket_map: dict[int, list[int]] = {}
    for idx in indices:
      value = self.lengths[idx]
      bucket_id = 0 if value <= 0 else (value - 1) // self.bucket_width
      bucket_map.setdefault(bucket_id, []).append(idx)
    return self._yield_uniform_batches(bucket_map)

  def _yield_uniform_batches(
      self, bucket_map: dict[int, list[int]]
  ) -> Iterator[list[int]]:
    target = self.target_budget_per_batch
    max_chunks = self.max_examples_per_batch

    def _new_batch():
      return {'indices': [], 'token_total': 0, 'chunk_total': 0,
              'sealed': False}

    def _place(batch, example_idx: int, example_budget: int) -> bool:
      if max_chunks and batch['chunk_total'] + 1 > max_chunks:
        return False
      next_tokens = batch['token_total'] + example_budget
      # A single example larger than the whole budget still has to go
      # somewhere: it gets a batch of its own.
      force_accept = batch['chunk_total'] == 0 and example_budget > target
      if next_tokens > target and not force_accept:
        return False
      batch['indices'].append(example_idx)
      batch['token_total'] = next_tokens
      batch['chunk_total'] += 1
      if (max_chunks and batch['chunk_total'] >= max_chunks) or (
          batch['token_total'] >= target
      ):
        batch['sealed'] = True
      return True

    prealloc = [_new_batch() for _ in range(self.estimated_batches)]
    extra: list[dict] = []
    cursor = 0

    def _assign_prealloc(example_idx: int, example_budget: int) -> bool:
      nonlocal cursor
      if not prealloc:
        return False
      for _ in range(len(prealloc)):
        batch = prealloc[cursor % len(prealloc)]
        cursor += 1
        if batch['sealed']:
          continue
        if _place(batch, example_idx, example_budget):
          return True
      return False

    def _assign_extra(example_idx: int, example_budget: int) -> None:
      while True:
        if not extra or extra[-1]['sealed']:
          extra.append(_new_batch())
        if _place(extra[-1], example_idx, example_budget):
          return
        extra[-1]['sealed'] = True

    for bucket_id in sorted(bucket_map.keys(), reverse=True):
      for example_idx in bucket_map[bucket_id]:
        budget = self.budget_lengths[example_idx]
        if not _assign_prealloc(example_idx, budget):
          _assign_extra(example_idx, budget)

    for batch in prealloc:
      if batch['indices']:
        yield batch['indices']
    for batch in extra:
      if batch['indices']:
        yield batch['indices']


def batch_iterator(
    streams: Sequence[Stream],
    batch_size: int,
    *,
    shuffle: bool = True,
    seed: int = 42,
    num_epochs: Optional[int] = None,
    drop_remainder: bool = True,
    pad_to_sentences: Optional[int] = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
  """Yield `(sentences, masks, lengths)` batches of documents."""
  if not streams:
    raise ValueError('batch_iterator requires at least one stream')
  rng = np.random.default_rng(seed)
  epoch = 0
  while num_epochs is None or epoch < num_epochs:
    order = np.arange(len(streams))
    if shuffle:
      rng.shuffle(order)
    for start in range(0, len(order), batch_size):
      idx = order[start:start + batch_size]
      if drop_remainder and len(idx) < batch_size:
        continue
      yield make_batch([streams[i] for i in idx], pad_to_sentences)
    epoch += 1


def make_token_budget_sampler(
    streams: Sequence[Stream],
    *,
    target_tokens_per_batch: int,
    bucket_width: int = 20,
    batch_size: int = 16,
    seed: int = 42,
) -> LengthBucketBatchSampler:
  """Build the reference's token-budget sampler over `streams`."""
  counts = stream_token_counts(streams)
  return LengthBucketBatchSampler(
      lengths=counts,
      batch_size=batch_size,
      bucket_width=bucket_width,
      seed=seed,
      budget_lengths=counts,
      target_budget_per_batch=target_tokens_per_batch,
  )


def token_budget_batch_iterator(
    streams: Sequence[Stream],
    sampler: LengthBucketBatchSampler,
    *,
    num_epochs: Optional[int] = None,
    pad_to_sentences: Optional[int] = None,
    pad_to_batch: Optional[int] = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
  """Yield token-budget batches, reshuffled each epoch.

  Every stream is emitted exactly once per epoch -- unlike the fixed-size
  iterator's `drop_remainder`, nothing is discarded, because short batches are
  padded out to `pad_to_batch` instead.
  """
  epoch = 0
  while num_epochs is None or epoch < num_epochs:
    sampler.set_epoch(epoch)
    for idx in sampler:
      yield make_batch(
          [streams[i] for i in idx], pad_to_sentences, pad_to_batch
      )
    epoch += 1
