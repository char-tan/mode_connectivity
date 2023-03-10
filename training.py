import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from typing import List, Optional, Callable, Dict
from argparse import Namespace

from utils.data import get_data_loaders
from utils.training_utils import train, test
from utils.utils import get_device
from training_config import TrainingConfig

from datetime import datetime


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

    if not training_config.weight_decay:
        training_config.weight_decay = 0

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
    elif training_config.lr_scheduler == "warmup_cosine":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=training_config.lr,
            epochs=training_config.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.02,
        )
        scheduler.step_frequency = "batch"
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
    tensorboard=False,
    profile=False,
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

    writer = (
        SummaryWriter(
            log_dir=f"./tensorboard/{datetime.now().strftime('%d%m%y_%H%M%S')}"
        )
        if tensorboard
        else None
    )

    for epoch in range(1, epochs + 1):
        train(
            args,
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            scheduler=scheduler,
            writer=writer,
            verbose=verbose,
            profile=profile,
        )
        test_loss, test_acc = test(model, device, test_loader, verbose=verbose)
        if writer:
            step = epoch * len(train_loader) + 1
            writer.add_scalar("loss/test", test_loss, global_step=step)
            writer.add_scalar("acc/test", test_acc, global_step=step)

        if scheduler:
            if scheduler.step_frequency == "epoch":
                scheduler.step()

    if writer:
        writer.close()

    return model
