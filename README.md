# A Comparative Study between Bayesian and Frequentist Neural Networks for Remaining Useful Life Estimation in Prognostics and Health Management

Official implementation of https://arxiv.org/abs/1911.06256. Bayesian and frequentist deep learning models for remaining useful life (RUL) estimation are evaluated on simulated run-to-failure data. Implemented in PyTorch, developed and tested on Ubuntu 18.04 LTS. All the experiments were run on a publicly available Google Compute Engine Deep Learning VM instance with 2 vCPUs, 13 GB RAM, 1 NVIDIA Tesla K80 GPU and *PyTorch 1.2 + fast.ai 1.0 (CUDA 10.0)* framework.

---------------------------------------------------------------------------------------------------------

## Requirements

Anaconda Python >= 3.6.4 (see https://www.anaconda.com/distribution/)

---------------------------------------------------------------------------------------------------------

## Installation

Clone or download the repository, open a terminal in the root directory and run the following commands:

```conda env create -f environment.yml```

```conda activate bayesian-deep-rul```

Now the virtual environment *bayesian-deep-rul* is active. To deactivate it, run:

```conda deactivate```

When you do not need it anymore, run the following command to remove it:

```conda remove --name bayesian-deep-rul --all```

---------------------------------------------------------------------------------------------------------

## Dataset

The models were tested on the four simulated turbofan engine degradation subsets in the publicly available *Commercial Modular Aero-Propulsion System Simulation* (C-MAPSS) dataset. Check *datasets/CMAPSS/README.md* for instructions on how to download the dataset.

---------------------------------------------------------------------------------------------------------

## Usage

Open a terminal in the root directory, activate the virtual environment and run one of the following commands:

*   `sh train.sh` to train the selected model. Parameters can be modified by editing *train.sh*

*   `sh evaluate.sh` to evaluate the selected model. Parameters can be modified by editing *evaluate.sh*

*   `sh run_experiments.sh` to replicate the experiments on the C-MAPSS dataset

---------------------------------------------------------------------------------------------------------

## TensorBoard

Open a terminal in the root directory, activate the virtual environment and run `tensorboard --logdir .` to monitor the training process with TensorBoard. If you are training on a remote server, connect through SSH and forward a port from the remote server to your local computer (`gcloud compute ssh <your-vm-name> --zone=<your-vm-zone> -- -L 6006:localhost:6006` on a Google Compute Engine Deep Learning VM instance).

---------------------------------------------------------------------------------------------------------

## Results

Training and evaluation logs of the experimental results are provided for verification. Run *results/results.ipynb* in Jupyter Notebook to check the results by yourself. TensorBoard logging was disabled to speed up training.

---------------------------------------------------------------------------------------------------------

## Citation

If you find this work useful in your research, please consider citing:

@article{dellalibera2019comparative,
    title={A Comparative Study between Bayesian and Frequentist Neural Networks for Remaining Useful Life Estimation in Condition-Based Maintenance},
    author={Luca Della Libera},
    year={2019},
    journal={arXiv preprint arXiv:1911.06256},
    eprint={1911.06256},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}

---------------------------------------------------------------------------------------------------------

## Contact

luca310795@gmail.com

---------------------------------------------------------------------------------------------------------
