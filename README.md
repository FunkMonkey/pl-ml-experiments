# Personalized Learning Experiments with Machine Learning

This repository contains the code for various ML experiments. It is structured as followed:

- `adlete_packages` - reusable code (e.g. configuration, data processing, models)
  - this may one day be extracted into an external repository
- `experiments` - code for running experiments (e.g. training runs etc.)

## More adaptive / personalized learning projects

See also [adlete - adaptive learning engine](https://gitlab.com/adaptive-learning-engine).

## Requirements

This repo uses [Pixi](https://pixi.sh) as a package manager.

## Installation

`pixi install`

## Running Experiments

The experiments generally consist of a combination of python scripts and `yaml` configuration files, e.g.:

- `src/experiments/knowledge_tracing/torch/dkt/train.py`
- `src/experiments/knowledge_tracing/torch/dkt/config.yaml`

You can generally simply run the experiment's python scripts directly, which will use the `yaml` configuration in the same folder by default. You can also specify a different configuration via `python train.py --config path/to/config.yaml`.

The datasets and metrics are currently saved in a `data` directory as a sibling directory of the repository.

### Viewing metrics in tensorboard

Generally the metrics are saved to the directory specified in the respective `yaml` configuration. Tensorboard can thus be started like this from the project root:

```bash
# opening the python environment via pixi
pixi shell
tensorboard --logdir=../data/training/dkt/lightning_logs/
```

### Validating yaml files using JSON schema

The `yaml` configuration files are validated using [pydantic](https://docs.pydantic.dev). If you use VS Code, you can also validate the files directly in the editor using the json-schema files that were created via `generate_json_schemas.py`. This requires the [yaml extension by redhat](https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml).

Currently the `yaml` files directly reference the json-schema files using relative filepaths. This will hopefully change soon.

## Current Experiments

- Deep Knowledge Tracing in Pytorch in `src/experiments/knowledge_tracing/torch/dkt`

## Upcoming changes

- web-hosting of the JSON schemas
- more experiments
