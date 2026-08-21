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
r"""Run the in-context reversal-curse probe against a trained TG checkpoint.

  python -m tg.inference.run_probe \
      --workdir /path/to/runs/tg_12m \
      --step best \
      --out probe.json

Reports mean target/distractor NLL per direction and the reverse-direction
margin (paper section 3.4). Restores through the same orbax manager the trainer
writes, so `--step best` selects the best-validation checkpoint -- which is the
one the reference reports.
"""

# pylint: disable=invalid-name,g-importing-member

import json
from absl import app
from absl import flags
from absl import logging

from tg.data import tg as tg_data
from tg.inference import probes
from tg.inference import tg as tg_infer
from tg.models import tg_config
from tg.models import tg_model


_WORKDIR = flags.DEFINE_string('workdir', None, 'Run directory.', required=True)
_STEP = flags.DEFINE_string(
    'step', 'best', "Checkpoint step, or 'best' / 'latest'.")
_OUT = flags.DEFINE_string('out', None, 'Optional JSON output path.')
_MAX_PAIRS = flags.DEFINE_integer('max_pairs', 100, 'Number of name pairs.')


def _load(workdir: str, step: str):
  """Restore params + config from a trainer-written checkpoint."""
  import orbax.checkpoint as ocp  # local: heavy import

  cfg_path = f'{workdir}/config.json'
  with open(cfg_path) as f:
    saved = json.load(f)
  cfg = tg_config.TgConfig(**saved['model'])

  mgr = ocp.CheckpointManager(f'{workdir}/checkpoints')
  if step == 'best':
    target = mgr.best_step()
    if target is None:
      target = mgr.latest_step()
  elif step == 'latest':
    target = mgr.latest_step()
  else:
    target = int(step)
  logging.info('restoring step %s from %s', target, workdir)
  restored = mgr.restore(target)
  params = restored['params'] if 'params' in restored else restored
  return cfg, params, target


def main(argv):
  del argv
  cfg, params, step = _load(_WORKDIR.value, _STEP.value)

  tokenizer = tg_data.build_tokenizer()
  specials = tg_data.special_ids_from_tokenizer(tokenizer)
  tg_infer.check_specials_match_cfg(specials, cfg)

  model = tg_model.ThoughtGestaltDo(cfg)
  summary = probes.run_probe(
      model.apply, params, cfg, specials, tokenizer,
      max_pairs=_MAX_PAIRS.value,
  )
  summary['checkpoint_step'] = int(step)
  summary['workdir'] = _WORKDIR.value

  print()
  print(probes.format_summary(summary))
  if _OUT.value:
    probes.write_summary(summary, _OUT.value)
    print(f'\nwrote {_OUT.value}')


if __name__ == '__main__':
  app.run(main)
