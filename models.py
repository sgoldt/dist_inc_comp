"""
Models used in our experiments on (cloned) CIFAR10.

For each model, we note what we want to test with this model.

Author  : Sebastian Goldt <sgoldt@sissa.it>
Date    : August 2022
Version : 0.1

"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models  # for Resnet18

class TwoLayer(nn.Module):
    """
    Simple two-layer fully-connected network.

    GOAL: baseline model, confirmed on MNIST by Maria's experiment.
    """

    def __init__(self, D, K, num_classes=10):
        """
        Parameters:
        -----------

        D : input dimension
        K : number of hidden neurons
        """
        super().__init__()
        self.fc1 = nn.Linear(D, K)
        self.fc2 = nn.Linear(K, num_classes)

    def forward(self, x):
        x = self.fc1(x.squeeze())
        x = F.relu(x)
        x = self.fc2(x)

        return x


class ConvNet(nn.Module):
    """
    Simple convolutional network with two convolutional and two fully-connected
    layers. Taken from the pyTorch tutorial:
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

    GOAL: Investigate the effect of convolutions - do they lead to a quicker
    deviation from the model trained on GP?

    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        if num_classes < 100:
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)
        else:
            self.fc2 = nn.Linear(120, 120)
            self.fc3 = nn.Linear(120, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Resnet18(torch.nn.Module):
    """
    Resnet18 with tiny modification due to Joost van Amersfoort (y0ast)

    https://github.com/y0ast/pytorch-snippets/tree/main/minimal_cifar

    GOAL: what do residual connections do? what happens if we really push
    performance?

    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.resnet = models.resnet18(pretrained=False, num_classes=num_classes)

        self.resnet.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet.maxpool = torch.nn.Identity()

    def forward(self, x):
        x = self.resnet(x)

        return x


class AlexNet(nn.Module):
    """
    AlexNet implementation as described in [1]


    [1] Alex Krizhevsky: One weird trick for parallelizing
    convolutional neural networks <https://arxiv.org/abs/1404.5997>

    """
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

