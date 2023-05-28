# SLIC-HF

Reproduction of [SLiC-HF: Sequence Likelihood Calibration with Human Feedback](https://arxiv.org/abs/2305.10425)

(Work in progress)

## Installation

```sh
# Install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.3.1-0-Linux-x86_64.sh
sh Miniconda3-py39_23.3.1-0-Linux-x86_64.sh

# Install this
source devops/install.sh
```

## Running

```sh
# Local development
sh src/do_everything_debug.sh

# Full training
sh src/do_everything_large.sh
```

## Tensorboard

* [Long short experiment](https://tensorboard.dev/experiment/BbA11fD1Rhq2OAszMjNUXw/#text&runSelectionState=eyJzbGljL2xvbmdfc2hvcnQvc2xpY19sb3NzXzFlLTA1XzE2ODUyOTY0NzMiOmZhbHNlfQ%3D%3D&_smoothingWeight=0.999) - Comparing `src/train/train_slic.py::slic_loss` function vs `src/train/train_slic.py::slic_loss_logits` function on `t5-base`.
