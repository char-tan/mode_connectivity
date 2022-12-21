from dataclasses import dataclass
from typing import List, Optional, Callable, Dict

from models.cnn import CNN
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
    lr_scheduler: Optional[str] = None
    log_interval: int = 10
    weight_decay: Optional[float] = None
    momentum: Optional[float] = 0.9

CNN_CIFAR10_DEFAULT = TrainingConfig(
    model_factory=CNN, 
    dataset="cifar10", 
    batch_size=512, 
    lr=0.001, 
    epochs=50,
    )

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
    model_kwargs={"depth": 22, "width_multiplier": 2},
    dataset="cifar10",
    batch_size=100,
    lr=0.001,
    epochs=100,
    )

VGG_CIFAR10_DEFAULT = TrainingConfig(
    model_factory=VGG,
    model_kwargs={"vgg_name": "VGG16"},
    dataset="cifar10",
    batch_size=512,
    lr=0.001,
    epochs=50,
    )
