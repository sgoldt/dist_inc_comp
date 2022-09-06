"""
Experiment to train various neural networks on CIFAR10 or on three different
clones:

- gpiso   : class-wise Gaussian mixture with correct means, cov = rescaled identity
- gp      : class-wise Gaussian mixture with correct means, covariance
- wgan    : Ensemble of Wasserstein GAN (dcGAN architecture of Radford et al.)
- cifar5m : CIFAR10 clone by Nakkiran et al.: images sampled from DDPM model of Ho et
            al., labels by an 98.5 correct BigTransfer model.

Author  : Sebastian Goldt <sgoldt@sissa.it>
Date    : August 2022
Version : 0.1
"""

from __future__ import print_function

import argparse

import numpy as np

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

import cdatasets  # The datasets defined for this experiment
import models  # The models defined for this experiment
import utils  # utility functions for these experiments

# GLOBAL CONSTANTS
DATASET_ROOT = "/u/s/sgoldt/datasets"
DATASETS = ["gpiso", "gp", "wgan", "cifar5m", "cifar10"]
MODELS = ["twolayer", "mlp", "convnet", "resnet18"]

# Default values for optimisation parameters
# Optimised for Resnet18 according to the recipe of Joost van Amersfoort (y0ast)
# https://github.com/y0ast/pytorch-snippets/tree/main/minimal_cifar
LR_DEFAULT = 0.05
WD_DEFAULT = 5e-4
MOM_DEFAULT = 0.9
BS_DEFAULT = 128


def evaluate(model, loader, device="cpu"):
    """Evaluates the given model on the dataset provided by the given loader.

    Parameters:
    -----------
    model :
        a pyTorch module that implements the forward method. The expectation is
        that the model provides the output of a softmax followed by a logarithm,
        for example as computed by the eponymous F.log_softmax
    loader : pyTorch data loader object
    device : string indicating the device on which we run.

    Returns:
    --------
    loss     : the negative log-likelihood loss
    accuracy : the classification accuracy

    """
    model.eval()

    loss = 0
    correct = 0

    for data, target in loader:
        with torch.no_grad():
            data = data.to(device)
            target = target.to(device)

            prediction = model(data)
            loss += F.nll_loss(prediction, target, reduction="sum")

            prediction = prediction.max(1)[1]
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    loss /= len(loader.dataset)
    accuracy = correct / len(loader.dataset)

    return loss, accuracy


def get_welcome_string(args):
    """
    Returns the header of the log file.

    Parameters:
    -----------
    args : argparse object with the parameters given to the program
    """
    msg = f"""# Distributions of increasing complexity
# Training model {args.model} on {args.trainon}
# Arguments: {str(args)}
# (0) epoch, (1) step, (2) train loss, (3) train accuracy, (4) test loss, (5) test accuracy,
# (6) cifar10 train loss, (7) cifar10 train accuracy, (8) cifar10 test loss, (9) cifar10 accuracy"""
    return msg


def main():
    parser = argparse.ArgumentParser()
    models_help = "which model to train? " + " | ".join(MODELS)
    parser.add_argument("--model", default="twolayer", help=models_help)
    trainon_help = "which dataset to train on? " + " | ".join(DATASETS)
    parser.add_argument("--trainon", default="cifar10", help=trainon_help)
    epochs_help = "number of epochs to train (default: 50)"
    parser.add_argument("--epochs", type=int, default=50, help=epochs_help)
    lr_help = f"learning rate (default: {LR_DEFAULT})"
    parser.add_argument("--lr", type=float, default=LR_DEFAULT, help=lr_help)
    mom_help = f"momentum (default: {MOM_DEFAULT})"
    parser.add_argument("--mom", type=float, default=MOM_DEFAULT, help=mom_help)
    wd_help = f"weight decay (default: {WD_DEFAULT})"
    parser.add_argument("--wd", type=float, default=WD_DEFAULT, help=wd_help)
    bs_help = f"mini-batch size (default: {BS_DEFAULT})"
    parser.add_argument("--bs", type=int, default=BS_DEFAULT, help=bs_help)
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    iseed_help = "random seed to initialise the network (default: 0)"
    parser.add_argument("--iseed", type=int, default=1, help=iseed_help)
    parser.add_argument("--device", help="device on which to train: cpu | cuda")
    # parser.add_argument("--debug", action="store_true", help="evaluate after each epoch")

    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

    width = 32  # image width

    # First set the seed to initialise the network
    torch.manual_seed(args.iseed)
    # initialise model.
    if args.model == "twolayer":
        model = models.TwoLayer(width ** 2, 512)  # using grayscale images!
    elif args.model == "mlp":
        model = models.MLP(width ** 2, 512)  # using grayscale images!
    elif args.model == "convnet":
        model = models.ConvNet()
    elif args.model == "resnet18":
        model = models.Resnet18()
    else:
        raise ValueError("models need to be one of " + ", ".join(MODELS))
    model = model.to(device)

    # Now set the seed to run the training
    torch.manual_seed(args.seed)

    # Define the necessary data transforms
    transform = dict()
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    if args.model in ["twolayer", "mlp"]:
        # Transform to grayscale, vectorise
        transform_list += [
            transforms.Grayscale(),
            utils.FlattenTransform(width ** 2),
        ]
    for mode in ["train", "test"]:
        transform[mode] = transforms.Compose(transform_list)

    # Resnet18 transforms
    if args.model == "resnet18":
        norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transform["train"] = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm,
            ]
        )
        transform["test"] = transforms.Compose([transforms.ToTensor(), norm,])

    # Load datasets
    cifar10_dataset = dict()
    # if necessary, load the appropriate clone
    clone_dataset = None if args.trainon == "cifar10" else dict()
    for train in [True, False]:
        mode = "train" if train else "test"

        # Load CIFAR10 in any case
        cifar10_dataset[mode] = datasets.CIFAR10(
            DATASET_ROOT, train=train, transform=transform[mode], download=True
        )

        if args.trainon in ["gpiso", "gp"]:
            clone_dataset[mode] = cdatasets.GaussianCIFAR10(
                cifar10_dataset[mode],
                isotropic=(args.trainon == "gpiso"),
                train=train,
                transform=transform[mode],
            )
        elif args.trainon == "wgan":
            clone_dataset[mode] = cdatasets.ClonedCIFAR10(
                "./wgan/", "cifar10_wgan_ngf64", train=train, transform=transform[mode],
            )
        elif args.trainon == "cifar5m":
            clone_dataset[mode] = cdatasets.ClonedCIFAR10(
                "./cifar5m/", "cifar5m", train=train, transform=transform[mode]
            )

    # Create data loaders
    cifar10_loader = dict()
    clone_loader = None if clone_dataset is None else dict()
    kwargs = {"num_workers": 2, "pin_memory": True} if device == "cuda" else {}
    for mode in ["train", "test"]:
        bs = args.bs if mode == "train" else 5_000
        cifar10_loader[mode] = torch.utils.data.DataLoader(
            cifar10_dataset[mode], batch_size=bs, shuffle=(mode == "train"), **kwargs
        )

        if clone_loader is not None:
            clone_loader[mode] = torch.utils.data.DataLoader(
                clone_dataset[mode], batch_size=bs, shuffle=(mode == "train"), **kwargs,
            )

    # Now select the "target" dataset on which to train
    target_loader = cifar10_loader if args.trainon == "cifar10" else clone_loader

    # Optimiser: vanilla SGD.
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd
    )
    scheduler = None
    if args.model == "resnet18":
        milestones = [25, 40]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1
        )

    num_steps = 0  # number of actual SGD steps
    log_total_num_steps = np.log10(args.epochs * len(target_loader["train"]))
    steps_to_print = np.round(np.logspace(0, log_total_num_steps, num=100, base=10))
    steps_to_print = list(np.unique(steps_to_print.astype(int)))
    # make sure we also evaluate at the end of training
    steps_to_print[-1] -= 1

    # set up logfile
    fname_root = f"dic_{args.model}_{args.trainon}_bs{args.bs}_lr{args.lr:g}_mom{args.mom:g}_wd{args.wd:g}_iseed{args.iseed}_seed{args.seed}"
    log_file = open(f"logs/{fname_root}.log", "w", buffering=1)
    welcome = get_welcome_string(args)
    utils.log(welcome, log_file)

    model.train()
    for epoch in range(1, args.epochs + 1):
        for data, target in target_loader["train"]:
            # Testing
            if num_steps == 0 or num_steps in steps_to_print:
                # It's useful to test at step 0, because if networks have same init,
                # they should have same performance on CIFAR10 no matter what they're
                # trained on at that time.
                model.eval()

                # Always evaluate on target data set first
                target_msg = ""
                for mode in ["train", "test"]:
                    loss, accuracy = evaluate(model, target_loader[mode], device)
                    target_msg += f"{loss:.4g}, {accuracy:.4g}, "

                # if training on clone, also evaluate on cifar10
                cifar10_msg = ""
                if args.trainon != "cifar10":
                    for mode in ["train", "test"]:
                        loss, accuracy = evaluate(model, cifar10_loader[mode], device)
                        cifar10_msg += f"{loss:.4g}, {accuracy:.4g}, "
                else:
                    # For consistency of log files, repeat loss and accuracy when training on CIFAR10
                    cifar10_msg = target_msg

                # remove trailing comma to avoid problems loading output
                msg = f"{epoch}, {num_steps}, " + target_msg + cifar10_msg[:-2]
                utils.log(msg, log_file)
                model.train()

            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            prediction = model(data)
            loss = F.nll_loss(prediction, target)

            loss.backward()
            optimizer.step()
            num_steps += 1

        if scheduler is not None:
            scheduler.step()

    torch.save(model.state_dict(), f"weights/{fname_root}_model.pt")


if __name__ == "__main__":
    main()
