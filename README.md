# DSAgentQA / CREDENCE

CREDENCE combines answers from retrieved sources by estimating their credibility and accounting for dependence between them. This repository contains the evidence-reading pipeline, the aggregation model and experiments on knowledge poisoning in retrieval-augmented generation.

## Code map

- `src/once/credence.py`: CREDENCE source assessment and explicit pooling.
- `src/once/`: isolated readings, evidence grids, model loading and related methods.
- `src/caer/`: additional evidential-fusion implementations.
- `experiments/`: data builders, reading-cache creation, training, attacks,
  baseline comparisons and metric summaries.

## Environment and inputs

Use Python 3.10 or newer and a suitable PyTorch/CUDA installation:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-optional.txt
```

Set the following paths before executing experiments:

| Variable | Contents |
| --- | --- |
| `ONCE_RAW` | Prepared benchmark and training records |
| `ONCE_CACHE` | Cached reader outputs |
| `ONCE_RUNS` | Trained model artifacts |
| `ONCE_RESULTS` | Evaluation outputs |
| `ONCE_MODEL_PATH` | Optional local reader-model override |

Data preparation is implemented in `experiments/build_*.py`. Reading-cache
construction uses `once_read.py` and `once_read_joint.py`. The cache model
identifier must match the identifier passed to the training runner; legacy
identifiers are preserved in the code for compatibility with existing caches.

## Main entry points

```bash
python experiments/build_train.py --help
python experiments/once_read.py --help
python experiments/once_read_joint.py --help
python experiments/credence_run.py --help
```

After creating matching training and evaluation caches:

```bash
python experiments/credence_run.py --tag main --seed 0 --device cuda:0
python experiments/credence_summary.py main
```

Use `--train`, `--model` and `--eval` to select the prepared cache families.
The runner includes controls for credibility, dependence, joint readings,
direct scoring and adversarial reading-state training.

## Outputs and checks

`credence_run.py` writes evaluation records to
`$ONCE_RESULTS/credence_<model>_<tag>_s<seed>.json`. The summary command groups
matching runs by tag. Keep cache identifiers and model settings consistent
between reading construction, training and evaluation.

To check the aggregation properties:

```bash
python experiments/test_properties.py
```

## Data and configuration

Download benchmark datasets and pretrained models from their original providers
and follow their usage terms. Data, checkpoints and generated predictions are
stored locally and are not included in this repository.

Set the paths described above before running experiments. Some auxiliary scripts
use fixed local path defaults; adjust those paths for your installation.
The command examples assume execution from the repository root.
