# pytorch-lr-dropout
This repo contains a PyTorch implementation of learning rate dropout from the paper "[Learning Rate Dropout](https://arxiv.org/abs/1912.00144)" by Lin et al.

To train a ResNet34 model on CIFAR-10 with the paper's hyperparameters, do

`python main.py --lr=.1 --lr_dropout_rate=0.5`

The original code is from the [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) repo. It uses [track-ml](https://github.com/richardliaw/track/tree/master/track) for logging metrics. This implementation doesn't add standard dropout.

## Preliminary results

The vanilla method is from`pytorch-cifar`: SGD with `lr=.1, momentum=.9, weight_decay=5e-4, batch_size=128`. The SGD-LRD method uses `lr_dropout_rate=0.5`. I ran four trials for each method.

![Alt text](images/lrd_learning_curve.png?raw=true "LRD")


It looks like LRD helps in the beginning of training, but does not provide major boosts after applying the LR schedule. Here are the final test accuracies:

| Method | This repo | Paper |
| ------ | ------------- | --------- |
| Vanilla | 95.45% | 95.30% |
| SGD-LRD |  94.43% | 95.54% |

## Official Implementation

Shorty after this repo was published, the authors created an official repo for their paper [here](https://github.com/HuangxingLin123/Learning-Rate-Dropout). The only differences I could find between the implementations are:
1. The official code uses `torch.bernoulli` for the mask while I use `(torch.rand_like(...) < lr_dropout_rate).type(d_p.dtype)`.
2. I use in-place elementwise-multiplication (`.mul_`) while they use `*`.
3. They clone `buf` before adding it to the parameters.
4. They multiply the LR and mask before adding it to the parameters, while I wait until the end and do `p.data.add_(-group["lr"], d_p)`.

It's unclear why these small differences would lead to such a large gap in performance between the implementations.
