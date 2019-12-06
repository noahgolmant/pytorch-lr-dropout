"""

"""
from enum import Enum
import torch
from torch.optim import Optimizer

SampleMode = Enum("SampleMode", "unif_cdf bernoulli")

## stolen from torch.optim.optimizer
class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()
####################################


class OfficialSGDLRD(Optimizer):
    """
    This is the learrning rate dropout implementation from the official repo:
    https://github.com/HuangxingLin123/Learning-Rate-Dropout/blob/master/cifar10/sgd_lrd.py
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        dropout=0.0,
        nesterov=False,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            dropout=dropout,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(OfficialSGDLRD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(OfficialSGDLRD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                ## mask
                m = torch.ones_like(p.data) * group["dropout"]
                mask = torch.bernoulli(m)

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(1 - dampening, d_p)

                ##dropout learning rate
                lr_dropout = group["lr"] * mask
                I_buf = lr_dropout * buf.clone()

                p.data.add_(-1, I_buf)

        return loss


class SGDLRD(Optimizer):
    r"""Implements learning rate dropout from the paper "Learning Rate Dropout"
        by Lin et. al (https://arxiv.org/abs/1912.00144)
        Original SGD implementation is taken from:
            https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        lr_dropout_rate (float, optional): Bernoulli parameter of binary mask
            for each update. Each update retained w.p. `lr_dropout_rate`
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = SGDLRD(model.parameters(), lr=0.1, lr_dropout_rate=0.5,
                               momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params,
        lr=required,
        lr_dropout_rate=0.0,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        sample_mode=SampleMode.unif_cdf,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if lr_dropout_rate < 0.0:
            raise ValueError(
                "Invalid learning rate dropout parameter: {}".format(lr_dropout_rate)
            )
        elif lr_dropout_rate == 0.0:
            raise ValueError(
                "Learning rate dropout must be positive in order to retain some entries"
            )
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.sample_mode = sample_mode

        defaults = dict(
            lr=lr,
            lr_dropout_rate=lr_dropout_rate,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDLRD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDLRD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            lr_dropout_rate = group["lr_dropout_rate"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p = d_p.add(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # construct random binary mask
                # each parameter retained with probability `lr_dropout_rate`
                mask = _get_mask(d_p, lr_dropout_rate, self.sample_mode)
                # apply the mask!
                d_p.mul_(mask)

                p.data.add_(-group["lr"], d_p)

        return loss


def _get_mask(data, dropout, sample_mode):
    device = data.get_device() if data.is_cuda else "cpu"
    if sample_mode == SampleMode.unif_cdf:
        mask = torch.rand_like(data, device=device, requires_grad=False) < dropout
    else:
        mask = torch.ones_like(data, device=device) * dropout
        mask = torch.bernoulli(mask)
    mask = mask.type(dtype=data.dtype)
    return mask
