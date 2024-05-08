# Deep4Life project: [Pigeons](https://www.bbc.com/news/science-environment-34878151) team

All the results are in the [Presentation](https://docs.google.com/presentation/d/1VP0hD3Spl1-d2TDUZf92NAkSbDK-Y4ACYHqFUOCzhMk/edit?usp=sharing).

## Installation

We recommend creating ```venv``` or ```coda``` environment with ```python>=3.9```. 

### Conda

```bash
conda create -n stellar python=3.9
source activate stellar
```

And then:
```bash
pip3 install -r requirements.txt
```

## Overview
When you want to run any experiment, run:
```bash
cd src
```
and then
```bash
python3 train_and_validate.py [ARGUMENTS]
```

with possible options:
  * ```-h, --help```: show this help message and exit
  * ```--dataset-path``` (default="data/train"): dataset path
  * ```--method``` {stellar,torch_mlp,sklearn_mlp,xgboost} (default="stellar"): 
  * ```--config``` (default="standard"): Name of a configuration in src/config/{method} directory.
  * ```--cv-seed``` (default=42): Seed used to make k folds for cross validation.
  * ```--n-folds``` (default=5): Number of folds in cross validation.
  * ```--retrain``` (default=True): Retrain a model using the whole dataset.
  


We recommend using ```--config``` flag. Sample configs are given in ```src/config/{method}``` folders.

### Notebooks
All the notebooks are in folder ```notebooks```. 

For each notebook run ```git config --local core.hooksPath .githooks/``` to set up git hooks for the project. 

## EDA

EDA is in file ```notebooks/exploratory-data-analysis.ipynb```.

## Baselines

We have 3 (4) baselines in total: 

### SVM

### XGBoost

Model is available in ```src/models/xgboost.py```. 
Experiments were ran using configs from ```src/config/xgboost/standard.yaml``` and notebook ```notebook/xgboost_tryout.ipynb```.

### MLP

There are two available models: one using **sklearn**, and one using **torch**. Models are implemented in ```src/models/{torch, sklearn}_mlp.py```.
Experiments were ran using configs from ```src/config/{torch, sklearn}_mlp/standard.yaml``` and notebooks ```notebook/MLP_tryout.ipynb```.

### CellSighter

## Stellar

Originally STELLAR was developed by [Snap Stanford](http://snap.stanford.edu/stellar).

PyTorch implementation of STELLAR, a geometric deep learning tool for cell-type discovery and identification in spatially resolved single-cell datasets. STELLAR takes as input annotated reference spatial single-cell dataset in which cells are assigned to their cell types, and unannotated spatial dataset in which cell types are unknown. STELLAR then generates annotations for the unannotated dataset. For a detailed description of the algorithm, please see our manuscript [Annotation of Spatially Resolved Single-cell Data with STELLAR](https://www.biorxiv.org/content/10.1101/2021.11.24.469947v2.full.pdf).


<p align="center">
<img src="https://github.com/snap-stanford/stellar/blob/main/images/stellar_overview.png" width="1100" align="center">
</p>

### Installation


**1. Python environment (Optional):**
We recommend using Conda package manager

```bash
conda create -n stellar python=3.8
source activate stellar
```

**2. Pytorch:**
Install [PyTorch](https://pytorch.org/). 
We have verified under PyTorch 1.9.1. For example:
```bash
conda install pytorch cudatoolkit=11.3 -c pytorch
```

**3. Pytorch Geometric:**
Install [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), 
follow their instructions. We have verified under Pyg 2.0. For example:
```bash
conda install pyg -c pyg
```

**4. Other dependencies:**

Please run the following command to install additional packages that are provided in [requirements.txt](https://github.com/snap-stanford/stellar/blob/main/requirements.txt).
```bash
pip install -r requirements.txt
```



