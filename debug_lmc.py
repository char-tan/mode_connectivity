from lmc import linear_mode_connect
from models.mlp import MLP
from models.vgg import VGG

linear_mode_connect(VGG, None, None, "mnist", n_points = 3)
