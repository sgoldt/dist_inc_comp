"""
Experiment to train various neural networks on CIFAR10/CIFAR100 data sets
or on various clones:

- gpiso   : class-wise Gaussian mixture with correct means, cov = rescaled identity
- gp      : class-wise Gaussian mixture with correct means, covariance
- gpc     : Gaussian mixture with correct means, covariance on the full data set with
            coarse-grained labels (hence more Gaussians than labels!)
- wgan    : Ensemble of Wasserstein GAN (dcGAN architecture of Radford et al.)
- cifar5m : CIFAR10 clone by Nakkiran et al.: images sampled from DDPM model of Ho et
            al., labels by an 98.5 correct BigTransfer model.

Author  : Sebastian Goldt <sgoldt@sissa.it>
Date    : August 2022
Version : 0.1
"""

from __future__ import print_function

import argparse
import os

import numpy as np

import timm  # advanced, pre-trained image models

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

import cdatasets  # The datasets defined for this experiment
import models  # The models defined for this experiment
from vit import ViT
import utils  # utility functions for these experiments

# GLOBAL CONSTANTS
DATASETS = ["cifar10", "cifar100", "cifar10c", "cifar100c"]
NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "cifar10c": 2, "cifar100c": 20}

CLONES = ["gpiso", "gp", "gpc", "wgan", "cifar5m"]
MODELS = [  # all the different architectures
    "twolayer",
    "convnet",
    "resnet18",
    "densenet121",
    "vit",
    "wide_resnet50_2",
]
# all the models that require vector inputs (and grayscale images)
vector_models = ["linear", "twolayer"]

# Default values for optimisation parameters, taken from the classic ImageNet paper
LR_DEFAULT = 0.05
WD_DEFAULT = 5e-4
MOM_DEFAULT = 0.9
BS_DEFAULT = 128


def evaluate(model, loader, loss_fn, device="cpu"):
    """Evaluates the given model on the dataset provided by the given loader.

    Parameters:
    -----------
    model :
        a pyTorch module that implements the forward method. The expectation is
        that the model provides the output of a softmax followed by a logarithm,
        for example as computed by the eponymous F.log_softmax
    loader : pyTorch data loader object
    loss_fn : callable that computes the loss
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
            loss += loss_fn(prediction, target)

            prediction = prediction.max(1)[1]
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    loss /= len(loader)
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
# Dataset: {args.dataset}, clone: {args.clone}, model {args.model}
# Arguments: {str(args)}
# (0) epoch, (1) step, (2) train loss, (3) train accuracy, (4) test loss, (5) test accuracy,
# (6) cifar train loss, (7) cifar train accuracy, (8) cifar test loss, (9) cifar accuracy"""
    return msg


def main():
    parser = argparse.ArgumentParser()
    models_help = "which model to train? " + " | ".join(MODELS)
    parser.add_argument("--model", default="twolayer", help=models_help)
    dataset_help = "what's the basic dataset? " + " | ".join(DATASETS)
    parser.add_argument("--dataset", default="cifar10", help=dataset_help)
    clone_help = "which clone to use for training? " + " | ".join(CLONES)
    parser.add_argument("--clone", default=None, help=clone_help)
    checkpoint_help = "checkpoint model weights"
    parser.add_argument("--checkpoint", action="store_true", help=checkpoint_help)
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
    parser.add_argument("--et", action="store_true", help="evaluate the train error")
    soft_help = "don't overwrite existing log files"
    parser.add_argument("--soft", action="store_true", help=soft_help)
    parser.add_argument("--dsetroot", default="~/datasets", help="Root for pytorch data sets")
    parser.add_argument("--dummy", action="store_true", help="dummy option")
    parser.add_argument(
        "--pretrained", action="store_true", help="evaluate the train error"
    )
    # parser.add_argument("--debug", action="store_true", help="evaluate after each epoch")

    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

    width = 32  # image width

    # First set the seed to initialise the network
    torch.manual_seed(args.iseed)
    # initialise model
    nc = NUM_CLASSES[args.dataset]
    if args.model == "twolayer":
        # using grayscale images!
        K = 512 if args.dataset == "cifar10" else 2048
        model = models.TwoLayer(width**2, K, nc)
    elif args.model == "convnet":
        model = models.ConvNet(nc)
    elif args.model in ["resnet18", "wide_resnet50_2"]:
        model = timm.create_model(args.model, pretrained=args.pretrained)

        # make initial convolutions downsample less
        model.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = torch.nn.Identity()

        # Bring it down to ten classes
        model.fc = torch.nn.Linear(model.fc.in_features, nc)
    elif args.model == "densenet121":
        model = timm.create_model("densenet121", pretrained=args.pretrained)

        # make initial convolutions downsample less
        model.features.conv0 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.features.pool0 = torch.nn.Identity()

        # Bring it down to ten classes, apply the softmax
        model.classifier = torch.nn.Linear(model.classifier.in_features, nc)
    elif args.model == "vit":
        # ViT for cifar10 with standard settings taken from
        # https://github.com/kentaroy47/vision-transformers-cifar10/blob/main/train_cifar10.py
        model = ViT(
            image_size=32,
            patch_size=4,
            num_classes=10,
            dim=512,
            depth=6,
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1,
        )
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
    if args.model in vector_models:
        # Transform to grayscale, vectorise
        transform_list += [
            transforms.Grayscale(),
            utils.FlattenTransform(width**2),
        ]
    for mode in ["train", "test"]:
        transform[mode] = transforms.Compose(transform_list)

    # Advanced transforms for convolutional networks
    if args.model in ["convnet", "resnet18", "densenet121", "wide_resnet50_2", "vit"]:
        norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transform["train"] = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm,
            ]
        )
        transform["test"] = transforms.Compose(
            [
                transforms.ToTensor(),
                norm,
            ]
        )

    # Load datasets
    cifar_dataset = dict()
    # if necessary, load the appropriate clone
    if args.clone is not None:
        if args.clone.lower() == "none":
            args.clone = None

    clone_dataset = None if args.clone is None else dict()
    # boolean flag to indicate whether or not we're using CIFAR10
    is_cifar10 = args.dataset in ["cifar10", "cifar10c"]
    is_coarse = args.dataset.endswith("c")

    for train in [True, False]:
        mode = "train" if train else "test"

        # Load original data set in any case
        dset_class = datasets.CIFAR10 if is_cifar10 else datasets.CIFAR100
        cifar_dataset[mode] = dset_class(
            args.dsetroot, train=train, transform=transform[mode], download=True
        )
        if is_coarse:
            # coarse grain the labels, from 100 classes down to 20
            cifar_dataset[mode].targets = utils.cifar_coarse(
                cifar_dataset[mode].targets, "cifar10" if is_cifar10 else "cifar100"
            )

        if args.clone in ["gpiso", "gp"]:
            clone_dataset[mode] = cdatasets.GaussianCIFAR(
                cifar_dataset[mode],
                isotropic=(args.clone == "gpiso"),
                train=train,
                transform=transform[mode],
            )
        elif args.clone == "wgan":
            clone_dataset[mode] = cdatasets.ClonedCIFAR(
                f"./{args.dataset}_wgan/",
                f"{args.dataset}_wgan_ngf64",
                train=train,
                transform=transform[mode],
            )
        elif args.clone == "cifar5m":
            clone_dataset[mode] = cdatasets.ClonedCIFAR(
                "./cifar10_cifar5m/", "cifar5m", train=train, transform=transform[mode]
            )
        elif args.clone == "gpc":
            # Load CIFAR10/100 again
            cifar = dset_class(
                args.dsetroot, train=train, transform=transform[mode], download=True
            )
            # Create GP clone based on full 100 classes
            clone_dataset[mode] = cdatasets.GaussianCIFAR(
                cifar,
                isotropic=False,
                train=train,
                transform=transform[mode],
            )
            # Now coarse grain the labels
            clone_dataset[mode].targets = utils.cifar_coarse(
                clone_dataset[mode].targets, "cifar10" if is_cifar10 else "cifar100"
            )

    # Create data loaders
    cifar_loader = dict()
    clone_loader = None if clone_dataset is None else dict()
    kwargs = {"num_workers": 2, "pin_memory": True} if device == "cuda" else {}
    for mode in ["train", "test"]:
        bs = args.bs if mode == "train" else 5_000
        cifar_loader[mode] = torch.utils.data.DataLoader(
            cifar_dataset[mode], batch_size=bs, shuffle=(mode == "train"), **kwargs
        )

        if clone_loader is not None:
            clone_loader[mode] = torch.utils.data.DataLoader(
                clone_dataset[mode],
                batch_size=bs,
                shuffle=(mode == "train"),
                **kwargs,
            )

    # Now select the "target" dataset on which to train
    target_loader = cifar_loader if clone_loader is None else clone_loader

    # Optimiser: vanilla SGD.
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    loss_fn = F.cross_entropy

    num_steps = 0  # number of actual SGD steps
    log_total_num_steps = np.log10(args.epochs * len(target_loader["train"]))
    steps_to_print = np.round(np.logspace(0, log_total_num_steps, num=100, base=10))
    steps_to_print = list(np.unique(steps_to_print.astype(int)))
    # make sure we also evaluate at the end of training
    steps_to_print[-1] -= 1

    # set up logfile
    clone_name = args.dataset if args.clone is None else args.clone
    model_name = args.model + ("_pretrained" if args.pretrained else "")
    fname_root = f"dic_{args.dataset}_{clone_name}_{model_name}_bs{args.bs}_lr{args.lr:g}_mom{args.mom:g}_wd{args.wd:g}_iseed{args.iseed}_seed{args.seed}"
    log_file_path = f"{fname_root}.log"
    # make sure that there doesn't already exist a logfile with this filename
    if args.soft and os.path.isfile(log_file_path):
        print(f"Logfile {fname_root} already exists, will exit now")
        return
    log_file = open(log_file_path, "w", buffering=1)
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
                if args.et:
                    loss, accuracy = evaluate(
                        model, target_loader["train"], loss_fn, device
                    )
                    target_msg += f"{loss:.4g}, {accuracy:.4g}, "
                else:
                    target_msg += "nan, nan, "
                loss, accuracy = evaluate(model, target_loader["test"], loss_fn, device)
                target_msg += f"{loss:.4g}, {accuracy:.4g}, "

                # if training on clone, also evaluate on cifar10
                cifar_msg = ""
                if args.clone is not None:
                    if args.et:
                        loss, accuracy = evaluate(
                            model, cifar_loader["train"], loss_fn, device
                        )
                        cifar_msg += f"{loss:.4g}, {accuracy:.4g}, "
                    else:
                        cifar_msg += "nan, nan, "
                    loss, accuracy = evaluate(
                        model, cifar_loader["test"], loss_fn, device
                    )
                    cifar_msg += f"{loss:.4g}, {accuracy:.4g}, "
                else:
                    # For consistency of log files, repeat loss and accuracy
                    # when training on CIFAR10/100
                    cifar_msg = target_msg

                # remove trailing comma to avoid problems loading output
                msg = f"{epoch}, {num_steps}, " + target_msg + cifar_msg[:-2]
                utils.log(msg, log_file)

                if args.checkpoint:
                    torch.save(
                        model.state_dict(),
                        f"weights/{fname_root}_model_step{num_steps}.pt",
                    )

                model.train()

            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            prediction = model(data)

            loss = loss_fn(prediction, target)

            loss.backward()
            optimizer.step()
            num_steps += 1

        if scheduler is not None:
            scheduler.step()

    if args.checkpoint:
        torch.save(model.state_dict(), f"weights/{fname_root}_model.pt")


if __name__ == "__main__":
    main()
