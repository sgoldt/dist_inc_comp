"""
Datasets used in our experiments on (cloned) CIFAR10/100.

Author  : Sebastian Goldt <sgoldt@sissa.it>
Date    : August 2022
Version : 0.1

"""

import os
from typing import Any, Callable, Optional, Tuple

import numpy as np

from PIL import Image

import torch
from torchvision.datasets import CIFAR10, VisionDataset

# Code to create Gaussian clones
import censoring


class ClonedCIFAR(VisionDataset):
    """
    Dataset interface for a cloned version of CIFAR10/100, usually obtained using a
    combination of generative model + classifier.

    The code is based on the CIFAR10 implementation of pyTorch:
    https://pytorch.org/vision/stable/_modules/torchvision/datasets/cifar.html#CIFAR10

    The CIFAR100 class is just a subclass of that class that redefines some constants.

    Args:
        root (string): Root directory of dataset where files exist
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(
        self,
        root: str,
        name: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set
        mode = "train" if self.train else "test"

        self.data = np.load(os.path.join(root, f"{name}_{mode}_xs.npy"))
        self.targets = np.load(os.path.join(root, f"{name}_{mode}_ys.npy"))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


class GaussianCIFAR(VisionDataset):
    """
    Dataset interface for a Gaussian clone of CIFAR10/100, using the censoring library.

    The code is based on the CIFAR10 implementation of pyTorch:
    https://pytorch.org/vision/stable/_modules/torchvision/datasets/cifar.html#CIFAR10

    Args:
        cifar_dataset (CIFAR10/CIFAR100): loaded CIFAR10/100 dataset
        isotropic (bool, optional) : if True, covariances of the Gaussians are isotropic.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    """

    def __init__(
        self,
        cifar_dataset: CIFAR10,  # CIFAR100 is a subclass of CIFAR100
        isotropic: bool = False,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(None, transform=transform, target_transform=target_transform)

        if cifar_dataset.train != train:
            raise ValueError("mismatch between CIFAR data set given and train flag")

        self.train = train  # training set or test set

        # extract inputs, labels
        cifar_xs = torch.tensor(cifar_dataset.data).float()
        cifar_ys = torch.tensor(cifar_dataset.targets)

        # create clone
        clone_xs, clone_ys = censoring.censor2d(
            cifar_xs, cifar_ys, isotropic=isotropic
        )

        self.targets = clone_ys.numpy()

        # clamp the inputs to have the right range
        clone_xs = torch.clamp(clone_xs, min=0, max=255)
        # transform to numpy and...
        clone_xs = np.round(clone_xs.numpy())
        # ...match datatype of CIFAR
        clone_xs = np.array(clone_xs, dtype=cifar_dataset.data.dtype)
        self.data = clone_xs

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"
