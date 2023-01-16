from matplotlib import pyplot as plt
import numpy as np
import torch
from geodesic_opt import compare_lmc_to_geodesic, optimise_for_geodesic, plot_lmc_geodesic_comparison_obj
from models.mlp import MLP
from models.resnet import ResNet
from models.vgg import VGG
from super import SuperModel
from utils.metrics import euclid_dist, index_distance, sqrt_JSD_loss
from utils.utils import get_device, AddGaussianNoise
from utils.utils import state_dict_to_numpy_array, distance_to_line, generate_orthogonal_basis, projection, lerp_vectors
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

def get_dataloaders(dataset, noise_var): # returns a (test, train) pair
    return data.get_data_loaders(
        dataset=dataset, train_kwargs=train_kwargs, test_kwargs=test_kwargs, additional_train_transforms = AddGaussianNoise(mean = 0, std = np.sqrt(noise_var)), eval_only=True
    )

mlp_config = (MLP, "mlp_mnist_model", "", "mnist")
resnet_config = lambda n : ((ResNet, {"width_multiplier": n}), "resnet_wm", str(n), "cifar10")
vgg_config = lambda n : ((VGG, {"width_multiplier": n}), "vgg_wm", str(n), "cifar10")
# see model_files folder for which numbers are valid

def make_super(config, n, permuted, noise_var):
    # if permuted=True, uses the permuted version of the weights
    # (it is assumed both permuted and non-permuted exist, this is not a given)
    model_factory, name, name_n, dataset_name = config
    weights_a, weights_b = load_weights(name + name_n, permuted)
    trainloader, testloader = get_dataloaders(dataset_name, noise_var)
    if isinstance(config[0], tuple):
        model_factory = lambda : config[0][0](**config[0][1])
    else:
        model_factory = lambda : config[0] 
    return SuperModel(model_factory, n, weights_a, weights_b).to(device), trainloader, testloader, model_factory

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

def snapshot_plots(snapshots, save_path, experiment_name):
    final_geodesic_weights = snapshots[-1]['weights']
    v_start = state_dict_to_numpy_array(final_geodesic_weights[0])
    v_end = state_dict_to_numpy_array(final_geodesic_weights[-1])

    # find furthest point away from line connecting first and last model in param space
    distances_to_line = [(distance_to_line(v_start, v_end, state_dict_to_numpy_array(weights))) for weights in final_geodesic_weights]
    furthest_point = final_geodesic_weights[np.array(distances_to_line).argmax()]

    # find plane defined by v_start, v_end, and furthest point from the line v_start -- v_end
    furthest_point_plane = generate_orthogonal_basis(v_start, v_end, state_dict_to_numpy_array(furthest_point))

    for snapshot_i in snapshots:
        intermed_geodesic_weights = snapshot_i['weights']

        # project all other points onto this plane
        furthest_projected_points = []
        for weights in intermed_geodesic_weights:
            vi = state_dict_to_numpy_array(weights)
            furthest_projected_points.append(projection(vi, furthest_point_plane))
        
        fig_i, ax_i = plt.subplots()

        straight_line_points = [
            lerp_vectors(lam, furthest_projected_points[0], furthest_projected_points[-1]) for lam in np.linspace(0, 1, len(final_geodesic_weights) + 1)
        ]

        ax_i.scatter(np.array(straight_line_points)[:,0],np.array(straight_line_points)[:,1], label='straight line')
        ax_i.scatter(np.array(furthest_projected_points)[:, 0], np.array(furthest_projected_points)[:, 1], label='projected points')
        ax_i.set_xlabel('x-coordinate on chosen plane', fontsize = 14)
        ax_i.set_ylabel('y-coordinate on chosen plane', fontsize = 14)
        ax_i.legend()

        epoch_id = snapshot_i['epoch_id']
        batch_id = snapshot_i['batch_id']
        fig_i.suptitle('Projected points for epoch ' + str(epoch_id) + ' and batch '+ str(batch_id))
        fig_i.savefig(save_path + experiment_name + 'snapshot_epoch_' + str(epoch_id) + '_batch_' + str(batch_id) + '.png')
        fig_i.show()


def run_experiment(
    config,
    n_points=25,
    permute=True,
    geodesic_opt_lr=1e-1,
    geodesic_opt_epochs=3,
    num_snapshots_per_epoch = 1, # plot projections of supermodel
    noise_var = 0,
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
        permute,
        noise_var
    )

    # find geodesic and track path_actions during GD
    path_action, sq_euc_dists, snapshots = optimise_for_geodesic(
         super_model,
         train_loader,
         lr = geodesic_opt_lr,
         verbose=1,
         num_epochs=geodesic_opt_epochs,
         n_snapshots_per_epoch = num_snapshots_per_epoch,
    )

    torch.save(path_action, save_path + experiment_name + '_path_action.pt')
    torch.save(sq_euc_dists, save_path + experiment_name + '_sq_euc_dists.pt')
    torch.save(snapshots, save_path + experiment_name + '_snapshots.pt')
    torch.save(super_model, save_path + experiment_name + '_super_model.pt')

    # plot snapshots
    snapshot_plots(snapshots, save_path, experiment_name)

    # plot path action in GD alg
    fig, ax = opt_plot(path_action, sq_euc_dists)
    fig.suptitle(experiment_name + " - geodesic optimisation")

    fig.savefig(save_path + experiment_name + "_gdopt_plot.png")

    fig.show()

    # training and test accuracy comparison for lmc and geodesic
    comparison = compare_lmc_to_geodesic(
        super_model,
        model_factory,
        (test_loader, train_loader),
        distance_metric=distance_metrics,
        verbose = 1,
    )

    torch.save(comparison, save_path + experiment_name + "_comparison_dict.pt")

    fig2, ax2 = plot_lmc_geodesic_comparison_obj(
        comparison,
        figsize=plot_figsize,
        relative_x=plot_relative_x
    )

    fig2.suptitle(experiment_name + " - loss barrier")

    fig2.savefig(save_path + experiment_name + "_loss_barrier.png")

    fig2.show()

    return comparison, fig, fig2


