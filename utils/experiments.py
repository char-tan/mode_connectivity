from torchvision import transforms
import torch

from dataclasses import dataclass
from typing import List, Optional, Callable, Dict
from argparse import Namespace

from .data import get_data_loaders
from .training import train, test

from ..models.cnn import CNN
from ..models.mlp import MLP
from ..models.resnet import ResNet
from ..models.vgg import VGG


@dataclass
class ExperimentConfig:
    model_factory: Callable
    dataset: str
    batch_size: int
    epochs: int
    lr: float
    seed: int = 1
    opt: str = "adam"
    model_kwargs: Optional[Dict] = None
    lr_scheduler: Optional[str] = None
    log_interval: int = 10
    weight_decay: Optional[float] = None
    momentum: Optional[float] = 0.9


CNN_CIFAR10_DEFAULT = ExperimentConfig(
    model_factory=CNN, dataset="cifar10", batch_size=512, lr=0.001, epochs=50
)
MLP_CIFAR10_DEFAULT = ExperimentConfig(
    model_factory=MLP, dataset="cifar10", batch_size=512, lr=0.001, epochs=50
)
MLP_MNIST_DEFAULT = ExperimentConfig(
    model_factory=MLP, dataset="mnist", batch_size=512, lr=0.001, epochs=50
)
RESNET_CIFAR10_DEFAULT = ExperimentConfig(
    model_factory=ResNet,
    model_kwargs={"depth": 22, "width_multiplier": 2},
    dataset="cifar10",
    batch_size=100,
    lr=0.001,
    epochs=100,
)
VGG_CIFAR10_DEFAULT = ExperimentConfig(
    model_factory=VGG,
    model_kwargs={"vgg_name": "VGG16"},
    dataset="cifar10",
    batch_size=512,
    lr=0.001,
    epochs=50,
)


def setup_experiment(
    experiment_config: ExperimentConfig,
    additional_train_transforms: Optional[List] = None,
    additional_test_transforms: Optional[List] = None,
):
    device, device_kwargs = get_device()

    torch.manual_seed(experiment_config.seed)

    train_kwargs = {"batch_size": experiment_config.batch_size}
    test_kwargs = {"batch_size": experiment_config.batch_size}

    train_kwargs.update(device_kwargs)
    test_kwargs.update(device_kwargs)

    train_loader, test_loader = get_data_loaders(
        experiment_config.dataset,
        train_kwargs,
        test_kwargs,
        additional_train_transforms,
        additional_test_transforms,
    )

    if not experiment_config.model_kwargs:
        experiment_config.model_kwargs = {}
    model = experiment_config.model_factory(**experiment_config.model_kwargs).to(device)
    if experiment_config.opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=experiment_config.lr)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            momentum=experiment_config.momentum,
            lr=experiment_config.lr,
            weight_decay=experiment_config.weight_decay,
        )

    if experiment_config.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, experiment_config.epochs
        )
    else:
        scheduler = None

    return (
        model,
        device,
        train_loader,
        test_loader,
        optimizer,
        experiment_config.epochs,
        scheduler,
        experiment_config.log_interval,
    )


def run_simple_experiment(
    model,
    device,
    train_loader,
    test_loader,
    optimizer,
    epochs,
    scheduler,
    log_interval,
    num_runs=1,
    verbose: int = 2,
):
    # Need to do this because the train function takes an ArgumentParser object
    args = Namespace(log_interval=log_interval)
    for run in range(num_runs):
        for epoch in range(1, epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch, True, verbose=verbose)
            test(model, device, test_loader, True, verbose=verbose)
            if scheduler:
                scheduler.step()


