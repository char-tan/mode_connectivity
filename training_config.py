# %%
import torch

from dataclasses import dataclass
from typing import List, Optional, Callable, Dict, Union

from models.mlp import MLP
from models.resnet import ResNet
from models.vgg import VGG


@dataclass
class TrainingConfig:
    model_factory: Callable
    dataset: str
    batch_size: int
    epochs: int
    lr: float
    seed: int = 1
    opt: str = "adam"
    model_kwargs: Optional[Dict] = None
    lr_scheduler: Optional[Union[str, Callable]] = None
    log_interval: int = 10
    weight_decay: Optional[float] = None
    momentum: Optional[float] = 0.9


MLP_CIFAR10_DEFAULT = TrainingConfig(
    model_factory=MLP,
    dataset="cifar10",
    batch_size=512,
    lr=0.001,
    epochs=50,
)

MLP_MNIST_DEFAULT = TrainingConfig(
    model_factory=MLP,
    dataset="mnist",
    batch_size=512,
    lr=0.001,
    epochs=50,
)

RESNET_CIFAR10_DEFAULT = TrainingConfig(
    model_factory=ResNet,
    dataset="cifar10",
    batch_size=128,
    lr=0.1,
    epochs=250,
    weight_decay=1e-4,
    opt="sgd",
    lr_scheduler="warmup_cosine"
)

VGG_CIFAR10_DEFAULT = TrainingConfig(
    model_factory=VGG,
    dataset="cifar10",
    batch_size=128,
    lr=0.1,
    weight_decay=5e-4,
    epochs=50,
    opt="sgd",
    lr_scheduler="warmup_cosine"
)

# %%
