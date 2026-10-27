# DSAgentQA / CREDENCE

Code for credibility- and dependence-aware evidence aggregation in
retrieval-augmented generation.

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

- `ONCE_RAW`: prepared benchmark and training records.
- `ONCE_CACHE`: cached reader outputs.
- `ONCE_RUNS`: trained model artifacts.
- `ONCE_RESULTS`: evaluation outputs.
- `ONCE_MODEL_PATH`: optional local reader-model path override.

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

## Release contents

This repository contains the project source code, training and evaluation
entry points, baseline implementations, and available tests. Experiment and
metric-analysis code is retained. Manuscripts, downloaded reference papers,
editorial notes, publication figures and LaTeX-table exporters are not included.

Datasets, model weights, local prediction caches, logs, and machine credentials
are not distributed with this release. Obtain datasets and model weights from
their original providers and follow their respective access and usage terms.
Configuration defaults in some experimental scripts refer to the original
Windows or Linux workspace; use the documented environment variables and
command-line options, and review path constants before a full run.

The release was checked for Python syntax and accidental inclusion of manuscript
files and embedded credentials. This publication check does not rerun GPU
training or certify that every experimental entry point is turnkey.

## Repository status

Public code repository: https://github.com/xkk9866/DSAgentQA

The public release is a code snapshot. Dataset and checkpoint availability is
separate from source-code availability.
