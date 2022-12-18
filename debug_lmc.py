from lmc import linear_mode_connect
from models.mlp import MLP

linear_mode_connect(MLP, '','', "mnist", n_points = 3)
