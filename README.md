# pytorch-lr-dropout
"Learning Rate Dropout" in PyTorch

This repo contains a PyTorch implementation of learning rate dropout from the paper "[Learning Rate Dropout](https://arxiv.org/abs/1912.00144)" by Lin et al.

To train a ResNet34 model on CIFAR-10 with the paper's hyperparameters, do

`python train.py --lr=.1 --lr_dropout_rate=0.5`

It uses [track-ml](https://github.com/richardliaw/track/tree/master/track) for logging metrics.

## (TODO) Preliminary results
