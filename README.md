# SLIC-HF

Reproduction of [SLiC-HF: Sequence Likelihood Calibration with Human Feedback](https://arxiv.org/abs/2305.10425)

(Work in progress)

## Installation

```sh
conda create -n slic python=3.9 -y
conda activate slic
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

## Running

```sh
# Local development
sh src/do_everything_debug.sh

# Full training
sh src/do_everything_large.sh
```
