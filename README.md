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
