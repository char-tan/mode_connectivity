from lmc import linear_mode_connect
from models.mlp import MLP
from models.vgg import VGG
from models.resnet import ResNet

linear_mode_connect(ResNet, None, None, "cifar10", n_points=3)

