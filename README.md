# CARBOOST
CARBOOST: Chimeric Antigen Receptor Biophysically Oriented Oprtimization for Synaptic Topology.

`carboost` is a Python toolkit for CAR design via optimizing the synaptic cleft geometry.

It includes:
- `carboost.receptor`: Receptor structure filtering and end-to-end distance calculations
- `carboost.synapse`: Inter-membrane distance density and phi-value estimators
- `carboost.utils`: Utilities for carboost
- `carboost.folding`: ColabFold wrapper utilities for structure generation workflows

Packaged resources are included under `carboost/resources/` and are installed with pip.

## Install

From local source:

```bash
pip install -e .
```

From GitHub:

```bash
pip install "git+https://github.com/tiwarylab/carboost.git"
```

## Quick check

```bash
python -c "import carboost; print(carboost.__version__)"
```

## Packaged resources

Resource files for CD8alpha hinge distributions are included in:

- `carboost/resources/CD8alpha/af2rave/`
- `carboost/resources/CD8alpha/rMSA_AF2/`

The default loaders in `carboost.utils.load_utils` resolve these packaged resources automatically:

- `load_af2rave_KDEs(...)`
- `load_rMSA_AF2_KDEs(...)`

## Notebooks

Two demonstration notebooks are provided:

- `CARBOOST_demo_no_colabfold_CD22target.ipynb`
- `CARBOOST_toolkit_demo_with_colabfold_and_rMSA_AF2.ipynb`

They demonstrate end-to-end workflow examples for CD22 and for any sequence.

## Repository layout

- `carboost/`: package source code.
- `carboost/resources/`: packaged resource distributions used by loaders.
- `data/`: sample/generated structure files used in demos.
- `img/`: figures used in notebooks.
- `env-mac.yml`: optional environment file.
