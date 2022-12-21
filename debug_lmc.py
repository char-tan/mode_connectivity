from lmc import linear_mode_connect
from models.mlp import MLP

linear_mode_connect(MLP, 'model_files/mlp_mnist_a.pt','model_files/mlp_mnist_b.pt', "mnist", n_points = 3)
