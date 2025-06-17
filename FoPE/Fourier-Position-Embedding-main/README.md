# Fourier-Position-Embedding

This repository contains the code for the paper "Fourier Position Embedding: Enhancing Attention’s Periodic Extension for Length Generalization".

Our code is totally based on OLMo(<https://github.com/allenai/OLMo>). The core code of FoPE and main setup is available in the appendix of our paper, thus you can also integrate FoPE directly into any other repository.

## Installation

To install from source, run the following commands:

```
cd Fourier-Position-Embedding
pip install -e .[all]
```

## Data Downloading

The data preprocessing has been completed by the authors of [OLMo](https://github.com/allenai/OLMo), and you only need to download the preprocessed data for training.

To download the C4 dataset, run the following script:

```
cd Fourier-Position-Embedding
bash commands/run_download.sh
```

If you want to modify the saving directory, please change the default ```--local_filepath``` in ```scripts/download_data.py```.

## Pre-training

The following script shows how to pre-train a 60M model on the C4 dataset:

```
cd Fourier-Position-Embedding
bash commands/run_pretrain.sh
```

The configs for other settings are available in the ```configs``` directory.

## Fine-tuning

The following script shows how to extrapolate a 60M model on the C4 dataset:

```
cd Fourier-Position-Embedding
bash commands/run_finetune.sh
```

You may modify the ```eval_interval``` and ```save_interval_unsharded``` in the configs to get more fine-grained checkpoints.
