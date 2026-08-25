# Thought Gestalt

A JAX/Flax implementation of the **Thought Gestalt (TG)** model — a recurrent
transformer that models language at two levels of abstraction: tokens and
sentence-level "thought" states
([arXiv:2512.25026](https://arxiv.org/abs/2512.25026)). 


Built on [NanoDO](https://github.com/google-deepmind/nanodo), which supplies the
underlying JAX training infrastructure.  This version differs slightly from the
version of the model described in ([arXiv:2512.25026](https://arxiv.org/abs/2512.25026)).  
It achieves comparable results to those reported in the paper when trained
with 12M text tokens.

## Install

Requires Python 3.10 or 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For GPU or TPU, install the matching JAX build first — see the
[JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

## Train

TG expects a plain-text or JSONL corpus. In WikiText-style text, articles are
detected by `= Title =` headings.

```bash
python tg/main.py \
  --config=tg/configs/tg_default.py \
  --config.workdir=/tmp/tg_run \
  --config.tg_data_path=/path/to/wiki.train \
  --config.tg_val_data_path=/path/to/wiki.valid
```

Progress is written to `workdir` for [TensorBoard](https://github.com/tensorflow/tensorboard):

```bash
tensorboard --logdir /tmp/tg_run
```

### Key defaults

| | |
|---|---|
| Model | 12 layers, `d_model` 768, 12 heads, SwiGLU |
| Blocks | alternating self / cross-attention (`S,C,S,C,…`) |
| Sentence | `[BOS]` + up to 64 tokens + `[EOS]`, padded to 66 |
| Working memory | 40 sentence vectors |
| `S_REP` | extracted at layer 6, L2-normalized |
| Optimizer | AdamW, peak LR 2.5e-4, cosine decay, linear warmup from 0 |
| Batching | token-budget bucketing, 20,000 supervised tokens per step |
| Stream curriculum | 30 sentences, +12 every 5 epochs |

Override any of these on the command line, e.g. `--config.model.D=1024`. Set
`--config.model.remat=True` to trade compute for memory on long streams.

## Experiments

Ablations, baseline comparisons and the reversal-curse probe are documented in
[EXPERIMENTS.md](EXPERIMENTS.md).

## Citation

```bibtex
@article{borazjanizadeh2025thoughtgestalt,
  title  = {Modeling Language as a Sequence of Thoughts},
  author = {Borazjanizadeh, Nasim and McClelland, James L.},
  year   = {2025},
  eprint = {2512.25026},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url    = {https://arxiv.org/abs/2512.25026},
}
```

## Acknowledgements

The training infrastructure and the baseline decoder derive from
[NanoDO](https://github.com/google-deepmind/nanodo) (Google DeepMind).

## License

Apache License 2.0 — see [LICENSE](LICENSE). Portions derive from NanoDO,
copyright 2024 DeepMind Technologies Limited, used under the same license.
