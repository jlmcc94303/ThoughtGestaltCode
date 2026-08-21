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
"""GPT-2 + sentence-boundary bias: flatten TG's sentences into a token stream.

A pure data transformation; the model is the unchanged baseline decoder. Take
the TG pipeline's sentences, drop `[PAD]`, keep `[BOS]`/`[EOS]` so boundaries
stay explicit, then concatenate and pack into fixed `context_size` blocks.

In the resulting stream `[BOS]` predicts the first lexical token, the last
lexical token predicts `[EOS]`, and `[EOS]` predicts the next `[BOS]`.

Loss and reported perplexity cover lexical tokens only, matching TG's protocol,
so the comparison does not hinge on predicting frequent boundary events;
`lexical_label_weights` builds that mask.
"""

# pylint: disable=invalid-name

import math
from typing import Any, Iterable, Iterator, Optional, Sequence

import numpy as np

from tg.data import tg as tg_data


def flatten_stream(
    ids_SxL: np.ndarray,
    mask_SxL: np.ndarray,
) -> list[int]:
  """One TG stream `[S, L]` -> a flat token list with padding removed.

  `mask_SxL` is 1 on real tokens (including `[BOS]`, `[EOS]`, `[EOD]`) and 0 on
  padding, so masking is exactly "drop the pad filler, keep the markers".
  """
  if ids_SxL.shape != mask_SxL.shape:
    raise ValueError(
        f'ids {ids_SxL.shape} and mask {mask_SxL.shape} must have equal shape'
    )
  if ids_SxL.size == 0:
    return []
  return [int(t) for t in ids_SxL[mask_SxL.astype(bool)]]


def flatten_document(
    sentence_token_ids: Sequence[Sequence[int]],
    specials: tg_data.SpecialIds,
    *,
    max_sentence_tokens: int = 64,
    doc_terminal_order: str = 'eod_eos',
) -> list[int]:
  """Sentence ids -> flat `[BOS] .. [EOS] [BOS] .. [EOD] [EOS]` stream.

  Routed through `tensorize_document` rather than reimplemented so the boundary
  markers, the left-truncation rule and the document terminals are byte-for-byte
  the ones TG trains on. The only difference from TG's input is that padding is
  removed and sentences are concatenated.
  """
  ids_SxL, mask_SxL = tg_data.tensorize_document(
      sentence_token_ids,
      specials,
      max_sentence_tokens=max_sentence_tokens,
      sentence_tail_len=1,
      doc_terminal_order=doc_terminal_order,
  )
  return flatten_stream(ids_SxL, mask_SxL)


def pack_blocks(
    token_streams: Iterable[Sequence[int]],
    context_size: int,
    *,
    drop_remainder: bool = True,
) -> Iterator[np.ndarray]:
  """Noam-pack flat streams into `context_size` blocks.

  Mirrors `tg/data/gpt2.py:_NoamPack`: documents are concatenated and cut
  on a fixed stride, so a block may span a document boundary. The trailing
  partial block is dropped by default, matching the GPT-2 path (and keeping
  every block the same static shape).
  """
  if context_size <= 0:
    raise ValueError(f'context_size must be > 0; got {context_size}')
  packed: list[int] = []
  for stream in token_streams:
    start = 0
    data = list(stream)
    while start < len(data):
      rem = data[start:]
      if len(packed) + len(rem) < context_size:
        packed.extend(rem)
        break
      take = context_size - len(packed)
      packed.extend(rem[:take])
      yield np.asarray(packed, dtype=np.int32)
      start += take
      packed = []
  if packed and not drop_remainder:
    out = np.zeros((context_size,), dtype=np.int32)
    out[: len(packed)] = packed
    yield out


def blocks_from_cached_corpus(
    docs: Sequence[Sequence[Sequence[int]]],
    specials: tg_data.SpecialIds,
    context_size: int,
    *,
    max_sentence_tokens: int = 64,
    min_sentences_per_document: int = 2,
    doc_terminal_order: str = 'eod_eos',
) -> np.ndarray:
  """Cached SaT corpus -> `[N, context_size]` blocks for the GPT-2 baseline.

  Reusing the same cache as the TG runs is what makes this an apples-to-apples
  comparison: identical sentence boundaries, identical tokenizer, identical
  document set -- only the model's view of the stream differs.
  """
  streams = (
      flatten_document(
          sentences,
          specials,
          max_sentence_tokens=max_sentence_tokens,
          doc_terminal_order=doc_terminal_order,
      )
      for sentences in docs
      if len(sentences) >= min_sentences_per_document
  )
  blocks = list(pack_blocks(streams, context_size))
  if not blocks:
    return np.zeros((0, context_size), dtype=np.int32)
  return np.stack(blocks, axis=0)


def lexical_label_weights(
    labels_BxL: np.ndarray,
    specials: tg_data.SpecialIds,
) -> np.ndarray:
  """1.0 on label positions that are lexical tokens, 0.0 on specials/pad.

  Section 3.1: all reported losses and perplexities exclude special-token label
  positions, for TG and every baseline alike. Applying this to the *labels*
  (not the inputs) is what makes the number comparable to TG's
  `*_no_special` metrics.
  """
  special = {specials.pad, specials.bos, specials.eos, specials.eod}
  keep = np.ones(labels_BxL.shape, dtype=np.float32)
  for tok in special:
    keep = np.where(labels_BxL == tok, 0.0, keep)
  return keep


def count_tokens(blocks: np.ndarray, specials: tg_data.SpecialIds) -> dict:
  """Token accounting for a packed corpus, for comparison against TG runs."""
  total = int(blocks.size)
  special = {specials.pad, specials.bos, specials.eos, specials.eod}
  n_special = int(sum(int((blocks == t).sum()) for t in special))
  return {
      'blocks': int(blocks.shape[0]),
      'tokens_with_special': total,
      'tokens_no_special': total - n_special,
      'special_tokens': n_special,
  }


# --------------------------------------------------------------------------- #
# Iteration for the standard (non-TG) trainer
# --------------------------------------------------------------------------- #
class BlockIterator:
  """Shuffled, epoch-repeating iterator over packed `[N, L]` blocks.

  Yields `in_BxL` int32 batches, the interface `tg/training/train.py`
  expects from `py_batched_tfds`. Position is a plain integer so the iterator
  state is `(epoch, cursor)` -- small enough to checkpoint as scalars, unlike a
  pygrain iterator.
  """

  def __init__(
      self,
      blocks: np.ndarray,
      batch_size: int,
      *,
      num_epochs: Optional[int] = None,
      shuffle: bool = True,
      seed: int = 0,
      drop_remainder: bool = True,
  ):
    if blocks.ndim != 2:
      raise ValueError(f'blocks must be [N, L]; got shape {blocks.shape}')
    if batch_size <= 0:
      raise ValueError(f'batch_size must be > 0; got {batch_size}')
    if blocks.shape[0] < batch_size and drop_remainder:
      raise ValueError(
          f'only {blocks.shape[0]} blocks for batch_size={batch_size}; '
          'the corpus is too small to form one batch'
      )
    self._blocks = blocks
    self._batch_size = int(batch_size)
    self._num_epochs = num_epochs
    self._shuffle = bool(shuffle)
    self._seed = int(seed)
    self._drop_remainder = bool(drop_remainder)
    self.epoch = 0
    self.cursor = 0
    self._order = self._order_for_epoch(0)

  def _order_for_epoch(self, epoch: int) -> np.ndarray:
    n = self._blocks.shape[0]
    if not self._shuffle:
      return np.arange(n)
    # Derived from (seed, epoch) so the order is reproducible after a restore
    # without serializing the permutation itself.
    return np.random.default_rng(self._seed + epoch).permutation(n)

  @property
  def steps_per_epoch(self) -> int:
    n = self._blocks.shape[0]
    if self._drop_remainder:
      return n // self._batch_size
    return int(math.ceil(n / self._batch_size))

  def get_state(self) -> dict:
    return {'epoch': int(self.epoch), 'cursor': int(self.cursor)}

  def set_state(self, state: dict) -> None:
    self.epoch = int(state['epoch'])
    self.cursor = int(state['cursor'])
    self._order = self._order_for_epoch(self.epoch)

  def __iter__(self) -> 'BlockIterator':
    return self

  def __next__(self) -> np.ndarray:
    end = self.cursor + self._batch_size
    if end > len(self._order):
      if self._drop_remainder or self.cursor >= len(self._order):
        self.epoch += 1
        if self._num_epochs is not None and self.epoch >= self._num_epochs:
          raise StopIteration
        self._order = self._order_for_epoch(self.epoch)
        self.cursor = 0
        end = self._batch_size
    idx = self._order[self.cursor : end]
    self.cursor = end
    batch = self._blocks[idx]
    if batch.shape[0] < self._batch_size and not self._drop_remainder:
      pad = np.zeros(
          (self._batch_size - batch.shape[0], batch.shape[1]), np.int32)
      batch = np.concatenate([batch, pad], axis=0)
    return batch.astype(np.int32)


def build_iterators(
    train_path: str,
    val_path: str,
    tokenizer: Any,
    specials: tg_data.SpecialIds,
    context_size: int,
    batch_size: int,
    *,
    eval_batch_size: Optional[int] = None,
    document_format: str = tg_data.WIKITEXT,
    max_sentence_tokens: int = 64,
    min_sentences_per_document: int = 2,
    num_epochs: Optional[int] = None,
    seed: int = 0,
) -> tuple['BlockIterator', 'BlockIterator', dict]:
  """Read two corpus files and return (train_iter, eval_iter, stats)."""
  def _blocks(path: str) -> np.ndarray:
    docs = []
    for raw in tg_data.read_documents(path, document_format):
      sentences = tg_data.split_sentences(raw)
      ids = tg_data.encode_sentences(
          sentences, tokenizer, max_sentence_tokens=max_sentence_tokens)
      if len(ids) >= min_sentences_per_document:
        docs.append(ids)
    return blocks_from_cached_corpus(
        docs, specials, context_size,
        max_sentence_tokens=max_sentence_tokens,
        min_sentences_per_document=min_sentences_per_document,
    )

  train_blocks = _blocks(train_path)
  val_blocks = _blocks(val_path)
  stats = {
      'train': count_tokens(train_blocks, specials),
      'val': count_tokens(val_blocks, specials),
  }
  train_it = BlockIterator(
      train_blocks, batch_size, num_epochs=num_epochs, shuffle=True, seed=seed)
  eval_it = BlockIterator(
      val_blocks, eval_batch_size or batch_size,
      num_epochs=1, shuffle=False, seed=seed)
  return train_it, eval_it, stats
