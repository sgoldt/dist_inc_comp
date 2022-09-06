"""
Utility functions for our experiments on learning distributions of
increasing complexity.

Author  : Sebastian Goldt <sgoldt@sissa.it>
Date    : August 2022
Version : 0.1

"""

import torch


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
