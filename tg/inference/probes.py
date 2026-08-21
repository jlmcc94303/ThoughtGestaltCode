

# pylint: disable=invalid-name,g-importing-member

import dataclasses
import json
from typing import Any, Callable, Optional, Sequence

import numpy as np

from tg.data import tg as tg_data
from tg.inference import tg as tg_infer
from tg.models.tg_config import TgConfig


NORMAL = 'normal'
REVERSED = 'reversed'
CONDITIONS = (NORMAL, REVERSED)

# The reference probe uses common first names so that each is a single GPT-2
# token; `build_pairs` filters to single-token names to keep "the first answer
# position" unambiguous.
DEFAULT_NAMES = (
    'John', 'Michael', 'David', 'Robert', 'James', 'William', 'Thomas',
    'Richard', 'Charles', 'Joseph', 'Daniel', 'Paul', 'Mark', 'George',
    'Kenneth', 'Steven', 'Edward', 'Brian', 'Ronald', 'Anthony',
)


@dataclasses.dataclass(frozen=True)
class ProbeExample:
  """One father-son item, in both directions."""

  father: str
  son: str

  @property
  def context(self) -> str:
    return f'The son of {self.father} is {self.son}.'

  def query(self, condition: str) -> str:
    if condition == NORMAL:
      return f'The son of {self.father} is'
    if condition == REVERSED:
      return f'The father of {self.son} is'
    raise ValueError(f'condition must be one of {CONDITIONS}; got {condition!r}')

  def target(self, condition: str) -> str:
    return self.son if condition == NORMAL else self.father

  def distractor(self, condition: str) -> str:
    """The other name in the prompt -- the order-bound shortcut answer."""
    return self.father if condition == NORMAL else self.son


@dataclasses.dataclass
class ProbeResult:
  condition: str
  target_nll: float
  distractor_nll: float

  @property
  def margin(self) -> float:
    """`delta`; negative means an in-context reversal error."""
    return self.distractor_nll - self.target_nll

  @property
  def is_reversal_error(self) -> bool:
    return self.margin < 0.0


def _single_token_id(name: str, tokenizer: Any) -> Optional[int]:
  """Token id for ' Name' if it is a single token, else None.

  The leading space matters: the answer position follows "is", so the target
  the model must produce is the space-prefixed form.
  """
  ids = list(tokenizer.encode(' ' + name, add_special_tokens=False))
  return int(ids[0]) if len(ids) == 1 else None


def build_pairs(
    tokenizer: Any,
    names: Sequence[str] = DEFAULT_NAMES,
    max_pairs: Optional[int] = None,
) -> list[ProbeExample]:
  """All ordered (father, son) pairs over names that are single tokens."""
  usable = [n for n in names if _single_token_id(n, tokenizer) is not None]
  pairs = [
      ProbeExample(father=a, son=b)
      for a in usable
      for b in usable
      if a != b
  ]
  if max_pairs is not None:
    pairs = pairs[: int(max_pairs)]
  return pairs


def _nll_from_logits(logits: np.ndarray, token_id: int) -> float:
  """-log softmax(logits)[token_id], computed stably."""
  shifted = logits - np.max(logits)
  return float(np.log(np.sum(np.exp(shifted))) - shifted[token_id])


def score_example(
    apply_fn: Callable[..., Any],
    params: Any,
    cfg: TgConfig,
    specials: tg_data.SpecialIds,
    tokenizer: Any,
    example: ProbeExample,
    condition: str,
) -> ProbeResult:
  """Score one item: run the context sentence, then read the answer position."""
  ctx_rows = tg_infer.encode_sentences(
      [example.context], tokenizer, specials, cfg,
      final_is_document_end=False,
  )
  query_ids = list(
      tokenizer.encode(example.query(condition), add_special_tokens=False)
  )
  logits = tg_infer.score_next_token(
      apply_fn, params, cfg, ctx_rows, query_ids, specials
  )
  tgt = _single_token_id(example.target(condition), tokenizer)
  dis = _single_token_id(example.distractor(condition), tokenizer)
  if tgt is None or dis is None:
    raise ValueError(
        f'names must be single tokens; got target={example.target(condition)!r}'
        f' distractor={example.distractor(condition)!r}'
    )
  return ProbeResult(
      condition=condition,
      target_nll=_nll_from_logits(logits, tgt),
      distractor_nll=_nll_from_logits(logits, dis),
  )


def run_probe(
    apply_fn: Callable[..., Any],
    params: Any,
    cfg: TgConfig,
    specials: tg_data.SpecialIds,
    tokenizer: Any,
    examples: Optional[Sequence[ProbeExample]] = None,
    max_pairs: int = 100,
) -> dict:
  """Run both conditions over every example and aggregate.

  Returns the four numbers Figure 5 plots -- mean target NLL and mean
  distractor NLL per condition -- plus the reverse-direction margin and the
  fraction of items that are outright reversal errors.
  """
  if examples is None:
    examples = build_pairs(tokenizer, max_pairs=max_pairs)
  if not examples:
    raise ValueError('no probe examples; check that names are single tokens')

  summary: dict = {'n_examples': len(examples)}
  for condition in CONDITIONS:
    results = [
        score_example(
            apply_fn, params, cfg, specials, tokenizer, ex, condition
        )
        for ex in examples
    ]
    margins = np.array([r.margin for r in results])
    summary[condition] = {
        'target_nll': float(np.mean([r.target_nll for r in results])),
        'distractor_nll': float(np.mean([r.distractor_nll for r in results])),
        'margin': float(np.mean(margins)),
        'margin_std': float(np.std(margins)),
        'reversal_error_rate': float(np.mean(margins < 0.0)),
    }
  return summary


def format_summary(summary: dict) -> str:
  """Human-readable table, mirroring how Figure 5 is read."""
  lines = [f'reversal-curse probe  (n = {summary["n_examples"]} pairs)', '']
  header = f'{"condition":<10} {"target":>9} {"distract":>9} {"margin":>9} {"err%":>7}'
  lines.append(header)
  lines.append('-' * len(header))
  for condition in CONDITIONS:
    s = summary[condition]
    lines.append(
        f'{condition:<10} {s["target_nll"]:>9.3f} {s["distractor_nll"]:>9.3f} '
        f'{s["margin"]:>9.3f} {100 * s["reversal_error_rate"]:>6.1f}%'
    )
  lines.append('')
  lines.append('margin = NLL(distractor) - NLL(target); < 0 is a reversal error')
  return '\n'.join(lines)


def write_summary(summary: dict, path: str) -> None:
  with open(path, 'w') as f:
    json.dump(summary, f, indent=2)
