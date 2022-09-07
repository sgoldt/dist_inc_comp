"""
Utility functions for our experiments on learning distributions of
increasing complexity.

Author  : Sebastian Goldt <sgoldt@sissa.it>
Date    : August 2022
Version : 0.1

"""

import numpy as np

import torch


# Animals vs objects is not balanced...
_CIFAR_COARSE_LABELS = {}
_CIFAR_COARSE_LABELS["cifar10"] = np.array([0, 0, 0, 0, 0,
                                            1, 1, 1, 1, 1])

# CIFAR100 coarse grained labels
# Thanks to Ryan Chan for this snippet
# (https://github.com/ryanchankh/cifar100coarse/blob/master/cifar100coarse.py)
_CIFAR_COARSE_LABELS["cifar100"] = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                              3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                              6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
                                              0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                              5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                             16,  4, 17,  4,  2,  0, 17,  4, 18, 17,
                                             10,  3,  2, 12, 12, 16, 12,  1,  9, 19, 
                                              2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                             16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                             18,  1,  2, 15,  6,  0, 17,  8, 14, 13])


def cifar_coarse(targets, dataset):
    """
    Coarse grains the given CIFAR100 labels to the 20 super classes.

    Parameters:
    -----------
    targets : a 1D array with the labels, as provided by CIFAR10/CIFAR100 data classes
    dataset : cifar10 | cifar100
    """
    return _CIFAR_COARSE_LABELS[dataset][targets]


def log(msg, logfile, print_to_out=True):
    """
    Print log message to  stdout and the given logfile.
    """
    logfile.write(msg + "\n")

    if print_to_out:
        print(msg)


class FlattenTransform(torch.nn.Module):
    """
    Convert image to 1D vector.  If the image is torch Tensor, it is expected to
    have [..., 1, H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Returns:
        PIL Image: Reshaped version of the input.

    """

    def __init__(self, D):
        super().__init__()
        self.D = D

    def forward(self, img):
        """
        Parameters:
        -----------
            img (PIL Image or Tensor): Image to be flattened

        Returns:
        --------
            PIL Image or Tensor: flattened image.
        """
        return img.reshape(-1, self.D)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.D})"
