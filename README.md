# pytorch-lr-dropout
This repo contains a PyTorch implementation of learning rate dropout from the paper "[Learning Rate Dropout](https://arxiv.org/abs/1912.00144)" by Lin et al. The official implementation can now be found [here](https://github.com/HuangxingLin123/Learning-Rate-Dropout).

To train a ResNet34 model on CIFAR-10 with the paper's hyperparameters, do

`python main.py --lr=.1 --lr_dropout_rate=0.5`

The original code is from the [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) repo. It uses [track-ml](https://github.com/richardliaw/track/tree/master/track) for logging metrics. This implementation doesn't add standard dropout.

## Preliminary results

The vanilla method is from`pytorch-cifar`: SGD with `LR=1, momentum=.9 weight_decay=5e-4, batch_size=128`. The SGD-LRD method uses `lr_dropout_rate=0.5`. I ran four trials for each method. When I first tried this, the official implementation wasn't available, so I used a different function to generate the random binary mask. It turns out that the two perform differently. Even though this appeared to be the only difference, the optimizer from the official repo performs significantly better (albeit with higher variance).

![Alt text](images/lrd_official.png?raw=true "LRD")

Here are the final test accuracies (averaged over four seeds):

| Method | This repo (original sampling) | This repo (`torch.bernoulli` sampling) | Paper (reported) | Paper (indendent run) |
| ------ | ------------- | --------- | -------- | ------- |
| Vanilla | 95.45% ± 0.07| - | 95.30% | - |
| SGD-LRD |  94.51% ± 0.22 | 94.36% ± 0.08 | 95.54% | 95.64% ± 0.22 |
