import argparse
import json
import os
import time

import numpy as np


def build_documents(path):
  """WikiText article split ('= Title ='), via the reference's title-scan."""
  from src_recurrent.pipelines.data.io import (
      extract_documents_from_text_by_title_rows,
  )
  docs = extract_documents_from_text_by_title_rows(path)
  out = []
  for d in docs:
    text = d.get('text') if isinstance(d, dict) else str(d)
    if text and text.strip():
      out.append(text)
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--input', required=True)
  ap.add_argument('--output', required=True)
  ap.add_argument('--splitter_model_name', default='sat-3l-sm')
  ap.add_argument('--sentence_threshold', type=float, default=0.15)
  ap.add_argument('--max_sentence_tokens', type=int, default=64)
  ap.add_argument('--min_sentence_tokens', type=int, default=3)
  ap.add_argument('--short_sentence_mode', default='merge')
  ap.add_argument('--limit_docs', type=int, default=0,
                  help='Process only the first N documents (for timing runs).')
  args = ap.parse_args()

  from src_recurrent.core.tokenizer_setup import build_tokenizer, tokenizer_signature
  from src_recurrent.core.data.sentence_splitter import (
      create_token_based_sentence_splitter,
  )

  tokenizer = build_tokenizer('gpt2')
  splitter = create_token_based_sentence_splitter(
      tokenizer=tokenizer,
      use_model=True,
      model_name=args.splitter_model_name,
      sentence_threshold=args.sentence_threshold,
      max_sentence_tokens=args.max_sentence_tokens,
      min_sentence_tokens=args.min_sentence_tokens,
      short_sentence_mode=args.short_sentence_mode,
  )
  if not splitter.use_model:
    raise RuntimeError(
        'SaT splitter failed to initialize; refusing to silently fall back to '
        'the regex splitter (that would defeat the point of this cache).'
    )

  docs = build_documents(args.input)
  if args.limit_docs:
    docs = docs[: args.limit_docs]
  print(f'documents: {len(docs)}', flush=True)

  tokens = []
  sent_offsets = [0]
  doc_sent_offsets = [0]
  t0 = time.time()
  n_sent = 0
  for i, text in enumerate(docs):
    sentences = splitter.split_text(text)
    for s in sentences:
      ids = tokenizer.encode(s, add_special_tokens=False)
      if not ids:
        continue
      if args.min_sentence_tokens and len(ids) < args.min_sentence_tokens:
        # The splitter already applies short_sentence_mode; this is a guard.
        continue
      ids = ids[: args.max_sentence_tokens]
      tokens.extend(ids)
      sent_offsets.append(len(tokens))
      n_sent += 1
    doc_sent_offsets.append(n_sent)
    if (i + 1) % 25 == 0 or i + 1 == len(docs):
      el = time.time() - t0
      rate = (i + 1) / max(el, 1e-9)
      eta = (len(docs) - i - 1) / max(rate, 1e-9)
      print(f'  {i+1}/{len(docs)} docs  {n_sent} sentences  '
            f'{len(tokens)} tokens  {el:.0f}s elapsed  ETA {eta/60:.1f} min',
            flush=True)

  os.makedirs(os.path.dirname(args.output), exist_ok=True)
  meta = {
      'input': args.input,
      'documents': len(docs),
      'sentences': n_sent,
      'tokens': len(tokens),
      'splitter_model_name': args.splitter_model_name,
      'sentence_threshold': args.sentence_threshold,
      'max_sentence_tokens': args.max_sentence_tokens,
      'min_sentence_tokens': args.min_sentence_tokens,
      'short_sentence_mode': args.short_sentence_mode,
      'use_model_splitter': True,
      'tokenizer_signature': tokenizer_signature(tokenizer),
      'vocab_size': len(tokenizer),
      'elapsed_sec': time.time() - t0,
  }
  np.savez_compressed(
      args.output,
      tokens=np.asarray(tokens, dtype=np.int32),
      sent_offsets=np.asarray(sent_offsets, dtype=np.int64),
      doc_sent_offsets=np.asarray(doc_sent_offsets, dtype=np.int64),
      meta=json.dumps(meta),
  )
  print(json.dumps(meta, indent=2))
  print(f'wrote {args.output}')


if __name__ == '__main__':
  main()
