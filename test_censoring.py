#!/usr/bin/env python3
"""
Test for the censoring utilities.

Author: Sebastian Goldt <goldt.sebastian@gmail.com>

June 2021
"""

import unittest

import torch
from torchvision import datasets

import censoring


class CensoringTests(unittest.TestCase):
    def test_covariances1d(self):
        D = 784

        dataset = datasets.MNIST("~/datasets/", train=False, download=True)
        xs = dataset.data.reshape(-1, D).float()
        ys = dataset.targets

        # Draw 10000 samples
        P = 10000

        # center the inputs to make test later easier
        xs -= torch.mean(xs)
        xs /= torch.std(xs)

        # obtain Gaussian samples
        gm_xs, gm_ys = censoring.censor1d(xs, ys, P)

        # Testing: evaluate the empirical covariance matrices of
        # MNIST samples and GM samples
        emp_cov_xx_gm = gm_xs.T @ gm_xs / P
        emp_cov_xy_gm = gm_xs.T @ gm_ys.float() / P
        emp_cov_xx = xs.T @ xs / P
        emp_cov_xy = xs.T @ ys.float() / P

        diff_xx = torch.sum((emp_cov_xx_gm - emp_cov_xx) ** 2)
        diff_xx /= torch.sum(emp_cov_xx ** 2)
        diff_xy = torch.sum((emp_cov_xy_gm - emp_cov_xy) ** 2)
        diff_xy /= torch.sum(emp_cov_xy ** 2)

        # Be nice here with the difference here.
        self.assertTrue(diff_xx < 1e-2, "input-input covariance incorrectly sampled")
        self.assertTrue(diff_xy < 1e-2, "input-label covariance incorrectly sampled")

    def test_covariances_isotropic1d(self):
        """
        Test specifically that *isotropic* covariances are estimated correctly.
        """
        D = 784

        dataset = datasets.MNIST("~/datasets/", train=True, download=True)
        xs = dataset.data.reshape(-1, D).float()
        ys = dataset.targets

        # Draw 10000 samples
        P = 10000

        # center the inputs to make test later easier
        xs -= torch.mean(xs)
        xs /= torch.std(xs)

        for isotropic in [True, False]:
            # obtain Gaussian samples
            gm_xs, gm_ys = censoring.censor1d(xs, ys, P, isotropic=isotropic)

            # Testing: evaluate the variance of inputs for each class
            # MNIST samples and GM samples
            for y in range(10):
                xs_i = xs[ys == y]
                gm_xs_i = gm_xs[gm_ys == y]

                var = torch.var(xs_i)
                gm_var = torch.var(gm_xs_i)

                self.assertTrue((var - gm_var) ** 2 / var ** 2 < 1e-2)

    def test_covariances2d_singlechannel(self):
        D = 784
        dataset = datasets.MNIST("~/datasets/", train=False, download=True)
        xs = dataset.data.float()
        ys = dataset.targets

        # Draw 10000 samples
        P = 10000

        # obtain Gaussian samples
        gm_xs, gm_ys = censoring.censor2d(xs, ys, P)

        # Center, reshape to compute moments
        xs = xs.reshape(-1, D)
        xs -= torch.mean(xs)
        xs /= torch.std(xs)

        gm_xs = gm_xs.reshape(-1, D)
        gm_xs -= torch.mean(gm_xs)
        gm_xs /= torch.std(gm_xs)

        # Testing: evaluate the empirical covariance matrices of
        # MNIST samples and GM samples
        emp_cov_xx_gm = gm_xs.T @ gm_xs / P
        emp_cov_xy_gm = gm_xs.T @ gm_ys.float() / P
        emp_cov_xx = xs.T @ xs / P
        emp_cov_xy = xs.T @ ys.float() / P

        diff_xx = torch.sum((emp_cov_xx_gm - emp_cov_xx) ** 2) / torch.sum(
            emp_cov_xx ** 2
        )
        diff_xy = torch.sum((emp_cov_xy_gm - emp_cov_xy) ** 2) / torch.sum(
            emp_cov_xy ** 2
        )

        # Be nice here with the difference here.
        self.assertTrue(diff_xx < 1e-2, "input-input covariance incorrectly sampled")
        self.assertTrue(diff_xy < 1e-2, "input-label covariance incorrectly sampled")

    def test_covariances2d_multichannel(self):
        """
        Censors CIFAR10, in all its three-channel glory.
        """
        width = 32
        num_channels = 3
        dataset = datasets.CIFAR10("~/datasets/", train=False, download=True)
        xs = torch.from_numpy(dataset.data).float()
        ys = torch.tensor(dataset.targets)

        # Draw 10000 Gaussian samples
        P = 10000
        gm_xs, gm_ys = censoring.censor2d(xs, ys, P)

        # Reshape from squares to vectors in each channel to compute moments
        xs = xs.reshape(-1, width ** 2, num_channels)
        gm_xs = gm_xs.reshape(-1, width ** 2, num_channels)

        for c in range(num_channels):
            xs_c = xs[:, :, c]
            gm_xs_c = gm_xs[:, :, c]

            xs_c -= torch.mean(xs_c)
            xs_c /= torch.std(xs_c)
            gm_xs_c -= torch.mean(gm_xs_c)
            gm_xs_c /= torch.std(gm_xs_c)

            # Testing: evaluate the empirical covariance matrices of
            # MNIST samples and GM samples
            emp_cov_xx_gm = gm_xs_c.T @ gm_xs_c / P
            emp_cov_xy_gm = gm_xs_c.T @ gm_ys.float() / P
            emp_cov_xx = xs_c.T @ xs_c / P
            emp_cov_xy = xs_c.T @ ys.float() / P

            diff_xx = torch.sum((emp_cov_xx_gm - emp_cov_xx) ** 2) / torch.sum(
                emp_cov_xx ** 2
            )
            diff_xy = torch.sum((emp_cov_xy_gm - emp_cov_xy) ** 2) / torch.sum(
                emp_cov_xy ** 2
            )

            self.assertTrue(diff_xx < 1e-2, "input-input cov incorrectly sampled")
            # Be nice here with the difference here. For CIFAR10, often the
            # difference in cov xy is < 1e-2, but not always
            self.assertTrue(diff_xy < 1e-1, "input-label cov incorrectly sampled")


if __name__ == "__main__":
    unittest.main()
