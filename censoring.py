"""
Small stand-alone utility to "censor" datasets, i.e. to estimate and sample from a
Gaussian mixture approximation of a given classification data set.

The two key methods are censor1d and censor2d, which compute censored versions of 1D
inputs and image datasets with one or more channels.

Date    : November 2020, August 2022 (image inputs)
Author  : Sebastian Goldt <goldt.sebastian@gmail.com>
Version : 0.2
"""

from scipy.stats import multivariate_normal

import torch


def censor1d(xs, ys, num_samples=None, isotropic=False):
    """Takes the given dataset consisting of D-dimensional vectors with class labels
    and creates a Gaussian mixture with matching covariance xx and xy.

    Parameters:
    -----------
    xs : (N, D)
        N inputs in D dimensions
    ys : (N)
        labels for K-class classification (numerical values 0,...,C-1)
    P : int
        number of samples to be drawn. If None, draw as many samples as there
        are in the original dataset.
    isotropic : bool
        if True, the covariance for each class is simply a scalar times the
        identity. Default is False.

    Returns:
    --------
    xs : (N, D)
        a set of N Gaussian vectors in D dimensions drawn from the Gaussian mixture
    ys : (N)
        the corresponding class labels
    """
    # If given, draw P samples
    num_samples = xs.shape[0] if num_samples is None else num_samples

    means, covs = _mixture_estimate1d(xs, ys, isotropic)

    gm_xs, gm_ys = _mixture_draw1d(means, covs, num_samples)

    return gm_xs, gm_ys


def _mixture_estimate1d(xs, ys, isotropic=False):
    """
    Estimates the means and covariances per class.

    Parameters:
    -----------
    xs : (N, D)
        N inputs in D dimensions
    ys : (N)
        labels for K-class classification (numerical values 0, ..., K-1)
    isotropic : bool
        if True, the covariance for each class is simply a scalar times the
        identity. Default is False.

    Returns:
    --------
    means : (num_classes, D)
        the means of the Gaussians in the mixture
    covs : (num_classes, D, D)
        the covariances of the Gaussians in the mixture
    """
    D = xs.shape[1]
    num_classes = ys.max() + 1
    # the vectorial mean for each cluster
    means = torch.zeros(num_classes, D)
    # The covariance matrix of each cluster
    covs = torch.zeros(num_classes, D, D)

    for i in range(num_classes):
        xs_i = xs[ys == i]  # extract the digits from that class
        means[i] = torch.mean(xs_i, axis=0)  # computer their mean...
        # ... and their covariance
        covs[i] = (xs_i - means[i]).T @ (xs_i - means[i]) / (xs_i.shape[0] - 1)

        if isotropic:
            covs[i] = torch.mean(torch.diag(covs[i])) * torch.eye(D)
        else:
            covs[i] += 1e-3 * torch.eye(D)  # for numerical stability

    return means, covs


def _mixture_draw1d(means, covs, num_samples=1):
    """
    Samples from the Gaussian mixture with the given means and covariances

    Parameters:
    --------
    means : (num_classes, D)
        the means of the Gaussians in the mixture
    covs : (num_classes, D, D)
        the covariances of the Gaussians in the mixture
    num_samples : int
        number of samples to be sampled
    """
    num_classes, D = means.shape

    # First the labels
    gm_ys = torch.randint(num_classes, (num_samples,))

    gm_xs = _mixture_drawWithLabels1d(means, covs, gm_ys)

    return gm_xs, gm_ys


def _mixture_drawWithLabels1d(means, covs, ys):
    """
    Samples inputs from a Gaussian mixture with the given means and covariances, where
    the inputs have the labels given by ys.

    Parameters:
    --------
    means : (num_classes, D)
        the means of the Gaussians in the mixture
    covs : (num_classes, D, D)
        the covariances of the Gaussians in the mixture
    ys : (num_samples)
        the labels of the samples to be sampled

    """
    num_classes, D = means.shape
    if ys.max() >= num_classes:
        raise ValueError("Received labels for more classes than I have Gaussians.")

    # Now the inputs
    gm_xs = torch.zeros(len(ys), D)

    for i in range(num_classes):
        indices = ys == i
        num_samples = int(indices.sum())

        gaussians = multivariate_normal.rvs(
            mean=means[i].numpy(), cov=covs[i].numpy(), size=num_samples
        )
        gm_xs[indices] = torch.from_numpy(gaussians).float()

    return gm_xs

def censor2d(xs, ys, num_samples=None, isotropic=False):
    """
    Takes the given dataset consisting of images with an arbitrary number of
    channels with class labels and creates a Gaussian mixture with
    matching covariance xx and xy.

    More precisely, we generate a Gaussian mixture with a Gaussian for each class and
    each channel. In each channel, the image is reshaped to a 1D array.

    Parameters:
    -----------
    xs : (N, L, L, C) or (N, L, L)
        N inputs of dimension LxL with C channels or just one channel.
    ys : (N)
        labels for K-class classification (numerical values 0, ..., K-1)
    P : int
        number of samples to be drawn. If None, draw as many samples as there
        are in the original dataset.
    isotropic : bool
        if True, the covariance for each class is simply a scalar times the
        identity. Default is False.

    Returns:
    --------
    xs : (N, L, L, C)
        a set of N Gaussian vectors in D dimensions drawn from the Gaussian mixture
    ys : (N)
        the corresponding class labels

    """
    if len(xs.shape) == 3:
        # add a an axis for the channels
        xs = xs[:, :, :, None]
    if len(xs.shape) != 4:
        raise ValueError(f"inputs must be of dimension (N, L, L, C) or  (N, L, L), got {xs.shape}")

    # If given, draw P samples
    num_samples = xs.shape[0] if num_samples is None else num_samples
    num_classes = ys.max() + 1
    _, width, _, num_channels = xs.shape

    # flatten the images to 1D vectors
    xs = xs.reshape(-1, width ** 2, num_channels)

    # First sample the labels
    gm_ys = torch.randint(num_classes, (num_samples,))
    # Container for the images, will be reshaped to square images at the end
    gm_xs = torch.zeros(num_samples, width ** 2, num_channels)

    # Now go through the channels
    for c in range(num_channels):
        # estimate mean and covariance for that channel
        means, covs = _mixture_estimate1d(xs[:, :, c], ys, isotropic)

        # means = torch.randn(*means.shape)

        # estimate inputs for that channel
        gm_xs[:, :, c] = _mixture_drawWithLabels1d(means, covs, gm_ys)

    # fold back
    gm_xs = gm_xs.reshape(-1, width, width, num_channels)

    return gm_xs, gm_ys
