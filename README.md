# pytorch-densenet
A pytorch implementation of DenseNet (https://arxiv.org/abs/1608.06993)

## Requirements

* Python 2.7
* Pytorch (newest version)

## Usage

This implementation currently supports training on CIFAR-10 dataset.

Simply run: 

```Shell
python train.py --help
```

which will print: 

```shell
usage: train.py [-h] [--gpu GPU] [--dataset DATASET] [--batch_size BATCH_SIZE]
                [--epoch EPOCH] [--seed SEED] [--checkpoint CHECKPOINT]
                [--resume_model RESUME_MODEL] [--lr LR] [--momentum MOMENTUM]

Densenet for Classification

optional arguments:
  -h, --help            show this help message and exit
  --gpu GPU
  --dataset DATASET
  --batch_size BATCH_SIZE
  --epoch EPOCH
  --seed SEED
  --checkpoint CHECKPOINT
  --resume_model RESUME_MODEL
  --lr LR
  --momentum MOMENTUM

```

