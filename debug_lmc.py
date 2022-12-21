from lmc import linear_mode_connect
from models.mlp import MLP
from models.vgg import VGG16

linear_mode_connect(VGG16, None, None, "mnist", n_points = 3)
