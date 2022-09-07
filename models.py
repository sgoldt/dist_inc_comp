"""
Models used in our experiments on (cloned) CIFAR10.

For each model, we note what we want to test with this model.

Author  : Sebastian Goldt <sgoldt@sissa.it>
Date    : August 2022
Version : 0.1

"""

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

        x = F.log_softmax(x, dim=1)

        return x


class MLP(nn.Module):
    """
    Simple rectangular fully-connected multi-layer perceptron.

    TODO: add batch norm to facilitate training - could of course impact effects.

    GOAL: investigate the impact of depth.
    """

    def __init__(self, D, K, num_classes):
        """
        Parameters:
        -----------

        D : input dimension
        K : number of hidden neurons
        """
        super().__init__()
        self.fc1 = nn.Linear(D, K)
        self.fc2 = nn.Linear(K, K)
        self.fc3 = nn.Linear(K, K)
        self.fc4 = nn.Linear(K, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x.squeeze()))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        x = F.log_softmax(x, dim=1)

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

        x = F.log_softmax(x, dim=1)

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

        x = F.log_softmax(x, dim=1)

        return x


class Resnet50(torch.nn.Module):
    """
    Resnet50

    GOAL: what does an interpolator do?

    """

    def __init__(self):
        super().__init__()

        self.resnet = models.resnet50(pretrained=False, num_classes=10)

    def forward(self, x):
        x = self.resnet(x)

        x = F.log_softmax(x, dim=1)

        return x
