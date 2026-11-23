# Self-teaching autoencoder

An autoencoder with no pixel loss and no pretrained teacher. A frozen copy of its own encoder judges whether a rebuilt photo "reads" the same as the original, on matching crops; four Wasserstein losses keep the latent from collapsing. Built on ideas from [this blog post](https://the-puzzler.github.io/blog/self-teaching-autoencoder/).

Part of a series of repos from my 2026 machine-learning sabbatical: a designer learning ML by building models from scratch and running experiments.

## What's here

- `MaskedAutoencoders/STL10-AE-working-self-teach.ipynb`: the 25M-parameter settings snapshot
- `models/*/README.md`: notes for the two exported 200k-step models

## Running it

**Data:** set `ML_DATASETS` to the folder that holds the datasets (torchvision downloads STL-10 / MNIST there). Paths like `<home>` and `<data-root>` are placeholders for my machines.

## Results

STL-10 test MSE 0.00467 for the 11M model (`ae_11M_selfteach_linskip_200k`), never trained on pixel MSE. Scaling to 25M needed small init, branch-output RMSNorm, per-token norms and 1,024 Wasserstein slices to stay stable.

**Code:** written by hand in Jupyter notebooks, with a chatbot as a tutor for questions and plotting helpers.

**Requirements:** Python 3.11+, PyTorch 2.x, torchvision, matplotlib, a CUDA GPU for anything beyond MNIST.
