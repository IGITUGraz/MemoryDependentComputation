# Memory-enriched computation and learning in spiking neural networks through Hebbian plasticity
This is the official code for the paper "Memory-enriched computation and learning in spiking neural networks through
Hebbian plasticity" ([[Abstract] arxiv.org](https://arxiv.org/abs/2205.11276),
[[PDF] arxiv.org](https://arxiv.org/pdf/2205.11276.pdf)).

## Setup
You need [PyTorch](https://pytorch.org) to run this code. We tested it on PyTorch version 1.7.0.
Additional dependencies are listed in [environment.yml](environment.yml). If you use
[Conda](https://docs.conda.io/en/latest/), run
```bash
conda env create --file=environment.yml
```
to install the required packages and their dependencies.

## Usage
Below you will find a short description of each task presented in the paper and instructions on how to reproduce the
results.

### Memorizing associations
In this task we tested the ability of our model to one-shot memorize associations and to use these associations later
when needed. The task requires to form associations between random continuous-valued vectors and integer labels that
were sequentially presented to the network.

#### Training
To start training on the association task, run
```bash
python memorizing_associations_task.py [optional arguments]
```
Set the command line argument `--sequence_length` to set the number of vector-label pairs (in the paper we used
sequences ranging from 1 to 55). Here, the command line argument `--num_classes` should be set to the same value.

To test the out-of-distribution capability of our model in this task, set `--sequence_length` < `--num_classes` and then
test the model with 2 < `--sequence_length` <= `--num_classes`.

#### Testing
To evaluate a trained model on the test data set, run:
```bash
python memorizing_associations_task.py --resume='PATH_TO_CHECKPOINT_FILE' --evaluate [optional arguments]
```
The optional model specific arguments must be set to the same values as during training, otherwise an error is thrown.

#### Plotting
To plot the network activity and the model's output after training, run:
```bash
python plot_memorizing_associations_task.py --checkpoint_path='PATH_TO_CHECKPOINT_FILE' [optional arguments]
```
The optional model specific arguments must be set to the same values as during training, otherwise an error is thrown.

## Multiprocessing Distributed Data Parallel Training
Model training can be distributed across multiple GPUs and multiple nodes (see below for examples). You should always
use the NCCL backend for multiprocessing distributed training since it currently provides the best distributed training
performance.

### Single node, multiple GPUs:
```bash
python SOME_SCRIPT.py --dist-url 'tcp://127.0.0.1:FREE_PORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 [optional arguments]
```

### Multiple nodes:
Node 0:
```bash
python SOME_SCRIPT.py --dist-url 'tcp://IP_OF_NODE0:FREE_PORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 0 [optional arguments]
```

Node 1:
```bash
python SOME_SCRIPT.py --dist-url 'tcp://IP_OF_NODE0:FREE_PORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 1 [optional arguments]
```

## Reference
If you use this code or models in your research and find it helpful, please cite the following paper:
```
@article{limbacher2022memory,
  title={Memory-enriched computation and learning in spiking neural networks through Hebbian plasticity},
  author={Limbacher, Thomas and {\"O}zdenizci, Ozan and Legenstein, Robert},
  journal={arXiv preprint arXiv:2205.11276},
  year={2022}
}
```

## Acknowledgments
Authors of this work are affiliated with Graz University of Technology, Institute of Theoretical Computer Science,
and Silicon Austria Labs, TU Graz - SAL Dependable Embedded Systems Lab, Graz, Austria. This work was supported by the
CHIST-ERA grant CHIST-ERA-18-ACAI-004, by the Austrian Science Fund (FWF) project number I 4670-N (project SMALL), and
by the "University SAL Labs" initiative of Silicon Austria Labs (SAL). We thank Wolfgang Maass and Arjun Rao for
initial discussions.