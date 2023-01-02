import torch
from torchvision import transforms

from typing import List, Optional, Callable, Dict
from argparse import Namespace

from .utils.data import get_data_loaders
from .utils.training_utils import train, test
from .utils.utils import get_device
from .training_config import TrainingConfig


def setup_train(
    training_config: TrainingConfig,
    additional_train_transforms: Optional[List] = None,
    additional_test_transforms: Optional[List] = None,
):

    device, device_kwargs = get_device()

    torch.manual_seed(training_config.seed)

    train_kwargs = {"batch_size": training_config.batch_size, "shuffle": True}
    test_kwargs = {"batch_size": training_config.batch_size, "shuffle": False}

    train_kwargs.update(device_kwargs)
    test_kwargs.update(device_kwargs)

    train_loader, test_loader = get_data_loaders(
        training_config.dataset,
        train_kwargs,
        test_kwargs,
        additional_train_transforms,
        additional_test_transforms,
    )

    if not training_config.model_kwargs:
        training_config.model_kwargs = {}
    model = training_config.model_factory(**training_config.model_kwargs).to(device)
    if training_config.opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=training_config.lr)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            momentum=training_config.momentum,
            lr=training_config.lr,
            weight_decay=training_config.weight_decay,
        )

    if training_config.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, training_config.epochs
        )
        scheduler.step_frequency = "epoch"
    elif training_config.lr_scheduler:
        scheduler = scheduler(optimizer)
        if not hasattr(scheduler, "step_frequency"):
            scheduler.step_frequency = "epoch"
    else:
        scheduler = None

    return (
        model,
        device,
        train_loader,
        test_loader,
        optimizer,
        training_config.epochs,
        scheduler,
        training_config.log_interval,
    )


def train_model(
    model,
    device,
    train_loader,
    test_loader,
    optimizer,
    epochs,
    scheduler,
    log_interval,
    verbose: int = 2,
):

    # Need to do this because the train function takes an ArgumentParser object
    args = Namespace(log_interval=log_interval)

    if scheduler is not None:
        if verbose == 2:
            scheduler.verbose = True

        if not hasattr(scheduler, "step_frequency"):
            print(
                "Warning: scheduler has no attribute step_frequency set. Defaulting to step_frequency='epoch'."
            )
            scheduler.step_frequency = "epoch"

    for epoch in range(1, epochs + 1):
        train(
            args,
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            scheduler=scheduler,
            verbose=verbose,
        )
        test(model, device, test_loader, verbose=verbose)
        if scheduler:
            if scheduler.step_frequency == "epoch":
                scheduler.step()

    return model
