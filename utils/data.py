import torch
from torchvision import transforms, datasets

from dataclasses import dataclass
from typing import List, Callable, Optional

@dataclass
class DatasetConfig:
    train_transforms: List
    test_transforms: List
    dataset: Callable


MNIST_CONFIG = DatasetConfig(
    train_transforms=[
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ],
    test_transforms=[transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))],
    dataset=datasets.MNIST,
)

CIFAR10_CONFIG = DatasetConfig(
    train_transforms=[
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ],
    test_transforms=[
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ],
    dataset=datasets.CIFAR10,
)

DATASETS_DICT = {"mnist": MNIST_CONFIG, "cifar10": CIFAR10_CONFIG}


def get_data_loaders(
    dataset: str,
    batch_size: int,
    additional_train_transforms: Optional[List] = None,
    additional_test_transforms: Optional[List] = None,
    root="./data",
):

    dataset_config = DATASETS_DICT[dataset]

    if not additional_test_transforms:
        additional_test_transforms = []
    if not additional_train_transforms:
        additional_train_transforms = []

    train_transforms = additional_train_transforms + dataset_config.train_transforms
    test_transforms = additional_test_transforms + dataset_config.test_transforms

    trainset = dataset_config.dataset(
        root=root, train=True, download=True, transform=train_transforms
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = dataset_config.dataset(
        root=root, train=False, download=True, transform=test_transforms
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader
