from matplotlib import pyplot as plt
import numpy as np
import torch
from geodesic_opt import compare_lmc_to_geodesic, optimise_for_geodesic, plot_lmc_geodesic_comparison_obj
from models.mlp import MLP
from models.resnet import ResNet
from models.vgg import VGG
from super import SuperModel
from utils.metrics import euclid_dist, index_distance, sqrt_JSD_loss
from utils.utils import get_device
from utils import data

device, _ = get_device()
path = "mode_connectivity/model_files/"
train_kwargs = {"batch_size": 128, 'num_workers': 0, 'pin_memory': False}
test_kwargs = {"batch_size": 128, 'num_workers': 0, 'pin_memory': False}
train_loader, test_loader = data.get_data_loaders(
    dataset="mnist", train_kwargs=train_kwargs, test_kwargs=test_kwargs, eval_only=True,
)

def load_weights(name, permuted):
    path1 = path + name + "_a.pt"
    path2_end = "_b_permuted.pt" if permuted else "_b.pt"
    path2 = path + name + path2_end
    return torch.load(path1, map_location=device), torch.load(path2, map_location=device)

def get_dataloaders(dataset): # returns a (test, train) pair
    return data.get_data_loaders(
        dataset=dataset, train_kwargs=train_kwargs, test_kwargs=test_kwargs, eval_only=True
    )

mlp_config = (MLP, "mlp_mnist_model", "", "mnist")
resnet_config = lambda n : (ResNet, "resnet_wm", str(n), "cifar10")
vgg_config = lambda n : (VGG, "vgg_wm", str(n), "cifar10")
# see model_files folder for which numbers are valid

def make_super(config, n, permuted):
    # if permuted=True, uses the permuted version of the weights
    # (it is assumed both permuted and non-permuted exist, this is not a given)
    model_factory, name, name_n, dataset_name = config
    weights_a, weights_b = load_weights(name + name_n, permuted)
    trainloader, testloader = get_dataloaders(dataset_name)
    return SuperModel(config[0], n, weights_a, weights_b).to(device), trainloader, testloader, model_factory

def rolling_mean(x, window):
    return np.convolve(x, np.ones(window), 'valid') / window

def opt_plot(path_lengths, sq_euc_dists, rolling_mean_length=50):
    fig, ax = plt.subplots()

    rolling_mean_length = rolling_mean(path_lengths, rolling_mean_length)

    loss_type = 'JSD loss'

    ax.plot(rolling_mean_length, color = 'red')
    ax.set_xlabel('GD iteration', fontsize = 14)
    ax.set_ylabel('path action via ' + loss_type, color = 'red', fontsize = 14)

    ax2=ax.twinx()
    ax2.plot(sq_euc_dists, color = 'blue')
    ax2.set_ylabel('path action via (squared) euclid dist', color = 'blue', fontsize = 14)
    fig.show()
    return fig, ax

def run_experiment(
    config,
    n_points=25,
    permute=True,
    geodesic_opt_lr=1e-1,
    geodesic_opt_epochs=3,
    plot_figsize=(10,5),
    plot_relative_x=True,
    distance_metrics={
        "index": index_distance,
        "sq.euclidean": euclid_dist,
        "JSD": sqrt_JSD_loss
    },
    save_path="mode_connectivity/experiments/",
    experiment_name="test"
):
    super_model, train_loader, test_loader, model_factory = make_super(
        config,
        n_points,
        permute
    )

    path_lengths, sq_euc_dists  = optimise_for_geodesic(
         super_model,
         train_loader,
         lr = geodesic_opt_lr,
         verbose=1,
         num_epochs=geodesic_opt_epochs
    )

    fig, ax = opt_plot(path_lengths, sq_euc_dists)
    fig.suptitle(experiment_name + " - geodesic optimisation")

    fig.savefig(save_path + experiment_name + "_gdopt_plot.png")

    fig.show()

    comparison = compare_lmc_to_geodesic(
        super_model,
        model_factory,
        (test_loader, train_loader),
        distance_metric=distance_metrics,
        verbose = 1,
    )

    torch.save(comparison, save_path + experiment_name + "_comparison_dict")

    fig2, ax2 = plot_lmc_geodesic_comparison_obj(
        comparison,
        figsize=plot_figsize,
        relative_x=plot_relative_x
    )

    fig2.suptitle(experiment_name + " - loss barrier")

    fig2.savefig(save_path + experiment_name + "_loss_barrier.png")

    fig2.show()

    return comparison, fig, fig2


