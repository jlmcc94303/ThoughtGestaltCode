# Experiments

Everything here is off by default and leaves the standard TG training path
unchanged. 

## Ablations

| Variant | Flag |
|---|---|
| Fixed token-span recurrence | `--config.model.segmentation=span --config.model.span_tokens=25` |
| In-context memory instead of cross-attention | `--config.model.memory_mode=in_context` |
| No working memory | `--config.model.use_memory=False` |
| Detached sentence representations | `--config.model.detach_sreps_for_memory=True` |
| Truncated backprop through memory | `--config.model.stm_backprop_window=8` |
| `S_REP` from the final layer | `--config.model.srep_extraction_layer=-1` |
| `S_REP` by mean-pooling content tokens | `--config.model.srep_pool_mode=mean_pool` |
| No context seeding | `--config.model.bos_replacement_mode=off` |
| μP parameterization | `--config.model.mup_enabled=True --config.model.mup_base_D=768` |
| Multiple `[EOS]` vectors per sentence | `--config.model.multi_eos_count=3` |

## Baseline comparisons

The repository carries a standard decoder-only transformer, which shares TG's
entry point, data pipeline, optimizer, evaluation and checkpoint format. Because
both read the same corpus and vocabulary, a comparison isolates the mechanism
under test rather than incidental differences in the harness.

```bash
# Plain decoder
python tg/main.py \
  --config=tg/configs/default.py \
  --config.workdir=/tmp/baseline_run \
  --config.vocab_path=/path/to/sentencepiece.model

# Decoder + sentence-boundary markers in the token stream
python tg/main.py --config=tg/configs/gpt2_boundary.py --config.workdir=...

# Decoder + gist attention masking
python tg/main.py --config=tg/configs/gpt2_gist.py --config.workdir=...
```

Model selection is a single dispatch in `tg/models/factory.py`, keyed on
`model_type`, returning a model paired with its loss function.

> The gist mask here is derived from the paper's description rather than from
> the original implementation, whose mask indexes the gist flag on the query
> axis instead of the key axis and so grants no cross-sentence access at all.
> Numbers from this config are therefore not expected to match published ones.

## Reversal-curse probe

Measures in-context relational-direction asymmetry: a relation is stated in the
prompt, then queried in the same direction or its inverse, and the likelihood of
the correct answer is compared against the other entity in the prompt.

```bash
python -m tg.inference.run_probe --workdir /tmp/tg_run --step best
```

Reports mean negative log-likelihood for the target and the distractor in each
direction, plus the reverse-direction margin.

