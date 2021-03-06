"""
Train CIFAR10 with PyTorch using "learning rate dropout" (https://arxiv.org/abs/1912.00144)
The rest of this code is based off of the excellent 'pytorch-cifar' repo:
    https://github.com/kuangliu/pytorch-cifar

Author: Noah Golmant
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from optimizer import SGDLRD
from utils import progress_bar

import track

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)

parser.add_argument(
    "--logroot", default="./logs", type=str, help="track-ml log directory"
)
parser.add_argument(
    "--dataroot", default="~/data", type=str, help="Download CIFAR data here"
)
parser.add_argument("--seed", default=0, type=int, help="pytorch random seed")

parser.add_argument(
    "--lr_dropout_rate",
    default=0.5,
    type=float,
    help="Bernoulli parameter for the random LR mask",
)


args = parser.parse_args()

torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print("==> Preparing data..")
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root=args.dataroot, train=True, download=True, transform=transform_train
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root=args.dataroot, train=False, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Model
print("==> Building model..")
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
net = ResNet34()
net = net.to(device)
if device == "cuda":
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    ckpt_path = os.path.join(track.trial_dir(), "ckpt.pth")
    checkpoint = torch.load(ckpt_path)
    net.load_state_dict(checkpoint["net"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]

criterion = nn.CrossEntropyLoss()
optimizer = SGDLRD(
    net.parameters(),
    lr=args.lr,
    lr_dropout_rate=args.lr_dropout_rate,
    momentum=0.9,
    weight_decay=5e-4,
)

lr_scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer=optimizer, milestones=[100, 150], gamma=0.1
)


# Training
def train(epoch):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_acc = 100.0 * correct / total

        progress_bar(
            batch_idx,
            len(trainloader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (train_loss / (batch_idx + 1), train_acc, correct, total),
        )
    lr_scheduler.step()
    train_loss = train_loss / len(trainloader)
    return train_loss, train_acc


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(testloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )

    # Save checkpoint.
    acc = 100.0 * correct / total
    if acc > best_acc:
        print("Saving..")
        state = {"net": net.state_dict(), "acc": acc, "epoch": epoch}
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        ckpt_path = os.path.join(track.trial_dir(), "ckpt.pth")
        torch.save(state, ckpt_path)
        best_acc = acc
    test_loss = test_loss / len(testloader)
    return test_loss, acc, best_acc


with track.trial(args.logroot, None, param_map=vars(args)):
    for epoch in range(start_epoch, start_epoch + 200):
        train_loss, train_acc = train(epoch)
        test_loss, test_acc, best_acc = test(epoch)
        track.metric(
            iteration=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            test_loss=test_loss,
            test_acc=test_acc,
            best_acc=best_acc,
        )
        track.debug(
            f"epoch {epoch} finished with stats: best_acc = {best_acc} | train_acc = {train_acc} | test_acc = {test_acc} | train_loss = {train_loss} | test_loss = {test_loss}"
        )
