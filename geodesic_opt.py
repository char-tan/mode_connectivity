# %%
import torch
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from evaluate import models_to_cumulative_distances, acc_on_path, evaluate_supermodel, evaluate_lmc
from super import SuperModel

from utils.metrics import JSD_loss, squared_euclid_dist, metric_path_length
from utils.objectives import heuristic_triplets, full_params
from utils.utils import lerp, get_device
from utils.training_utils import test
from utils.utils import load_checkpoint, intervals_to_cumulative_sums
from utils.metrics import JSD_loss
from lmc import model_interpolation

# ^ THIS DOES WORK AAAAAAA :)



def optimise_for_geodesic(
    super_model,
    dataloader,
    lr=0.01,
    num_epochs=1,
    n_snapshots_per_epoch = 1,
    verbose=1,
    loss_metric=JSD_loss,
):

    """
    pass in a super_model and dataloader, it will optimise this model for geodesic and return batch-wise path length and accuracy
    """

    device, _ = get_device()

    path_lengths = []
    sq_euc_dists = []

    n_interpolated = len(list(enumerate(dataloader)))
    snapshot_ids = np.round(np.linspace(0, n_interpolated - 1, n_snapshots_per_epoch)).astype(int)
    if (n_interpolated-1) not in snapshot_ids:
        snapshot_ids.append((n_interpolated -1))
    snapshots = []

    optimizer = torch.optim.SGD(super_model.parameters(), lr=lr)

    print("Optimising geodesic ...")
    # for _ in tqdm(range(max_iterations)):
    #     opt, loss, batch_images, data_iterator = objective_function(
    #         all_models, loss_metric, data_iterator, dataloader,
    #         device, learning_rate, n
    #     )

    for epoch_idx in range(num_epochs):
        if verbose > 0:
            print(f"Epoch {epoch_idx+1} of {num_epochs}")
        for batch_idx, (data, target) in tqdm(list(enumerate(dataloader))):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = super_model(data)

            path_length = metric_path_length(outputs, loss_metric=loss_metric)

            sq_euc_dist = super_model.sq_euc_dist()

            if verbose >= 2:
                print(
                    f"batch {batch_idx+1} | path length {path_length} | sq euc dist {sq_euc_dist}"
                )
            if batch_idx in snapshot_ids:
                snapshots.append({
                    'epoch_id': epoch_idx,
                    'batch_id': batch_idx,
                    'weights': [copy.deepcopy((super_model.models[i]).state_dict()) for i in range(len(super_model.models))]
                })

            path_length.backward()
            optimizer.step()

            path_lengths.append(path_length.item())
            sq_euc_dists.append(sq_euc_dist.item())

        if verbose == 1:
            print(
                f"epoch {epoch_idx+1} | path length {np.mean(path_lengths)} | sq euc dist {np.mean(sq_euc_dists)}"
            )

    return path_lengths, sq_euc_dists, snapshots

def compare_lmc_to_geodesic(
    geodesic_smodel,
    model_factory,
    dataloader,
    distance_metric,
    verbose = 1,
):
    if isinstance(dataloader, tuple):
        return tuple([
            compare_lmc_to_geodesic(
                geodesic_smodel,
                model_factory,
                specific_dataloader,
                distance_metric,
                verbose
            )
            for specific_dataloader in dataloader
        ])
    if isinstance(distance_metric, dict):
        distance_metrics = tuple(distance_metric.values())
    n_points = len(geodesic_smodel.models)
    weights_a = geodesic_smodel.models[0].state_dict()
    weights_b = geodesic_smodel.models[-1].state_dict()
    if verbose:
        print("Calculating LMC train accuracies ...")
    lmc_path_lengths, lmc_path_accs = evaluate_lmc(
        model_factory,
        n_points,
        weights_a,
        weights_b,
        dataloader,
        distance_metric=distance_metrics,
        verbose=1
    )

    if verbose:
        print("Calculating geodesic train accuracies ...")
    geodesic_path_lengths, geodesic_path_accs = evaluate_supermodel(
        geodesic_smodel,
        dataloader,
        distance_metrics,
        verbose
    )
    
    if isinstance(distance_metric, dict):
        lmc_dict = {
            # key is the name of a distance function in the dict
            # lmc_path_lengths[i] gets the values calculated for it
            key : lmc_path_lengths[i]
            for i, (key, value) in enumerate(distance_metric.items())
        }
        geodesic_dict = {
            key : geodesic_path_lengths[i]
            for i, (key, value) in enumerate(distance_metric.items())
        }
        return {
            "lmc": {
                "accuracies": lmc_path_accs,
                **lmc_dict
            },
            "geodesic": {
                "accuracies": geodesic_path_accs,
                **geodesic_dict
            }
        }        
    else:
        return {
            "lmc": {
                "lengths": lmc_path_lengths,
                "accuracies": lmc_path_accs
            },
            "geodesic": {
                "lengths": geodesic_path_lengths,
                "accuracies": geodesic_path_accs
            }
        }


def plot_lmc_geodesic_comparison_obj(
    comparison_obj, # either a single one, or pair (test, train)
    figsize=(10, 5),
    relative_x=False, # make x-axis from 0 to 1
):
    if not isinstance(comparison_obj, tuple) and "lengths" in comparison_obj["lmc"].keys():
        # Then we know this is using the "old-fashioned",
        # i.e. single-distance-returning, version of the
        # behaviour of compare_lmc_to_geodesic
        fig, ax = plt.subplots(figsize=figsize)
        lmc_xs = intervals_to_cumulative_sums(comparison_obj["lmc"]["lengths"])
        lmc_accs = comparison_obj["lmc"]["accuracies"]
        geodesic_xs = intervals_to_cumulative_sums(comparison_obj["geodesic"]["lengths"])
        geodesic_accs = comparison_obj["geodesic"]["accuracies"]
        ax.plot(lmc_xs, lmc_accs, label="lmc")
        ax.plot(geodesic_xs, geodesic_accs, label="geodesic")
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Distance along path by distance metric")
        ax.legend()
        fig.show()
        return fig, ax
    else:
        if not isinstance(comparison_obj, tuple):
            comparison_objs = (comparison_obj,)
        else:
            comparison_objs = comparison_obj
        distance_keys = list(comparison_objs[0]["lmc"].keys())
        distance_keys.remove("accuracies")
        num_dist_measures = len(distance_keys)
        fig, axs = plt.subplots(1, num_dist_measures, figsize=figsize, sharey=True)
        for i, dkey in enumerate(distance_keys):
            for j, obj in enumerate(comparison_objs):
                ax = axs[i]
                lmc_xs = intervals_to_cumulative_sums(obj["lmc"][dkey])
                lmc_accs = obj["lmc"]["accuracies"]
                geodesic_xs = intervals_to_cumulative_sums(obj["geodesic"][dkey])
                geodesic_accs = obj["geodesic"]["accuracies"]
                extra_lbl = ""
                if relative_x:
                    lmc_xs /= lmc_xs.max()
                    geodesic_xs /= geodesic_xs.max()
                if len(comparison_objs) == 2:
                    extra_lbl = ", test" if j == 0 else ", train"
                ax.plot(
                    lmc_xs,
                    lmc_accs,
                    label="lmc" + extra_lbl if i == 0 else None,
                    linestyle="solid" if j == 0 else "dashed",
                    # ^ solid for first (assumed test) dataloader, otherwise dashed,
                    marker='.'
                )
                ax.plot(
                    geodesic_xs,
                    geodesic_accs,
                    label="geodesic" + extra_lbl if i == 0 else None,
                    linestyle="solid" if j == 0 else "dashed",
                    marker='.'
                )
                if i == 0:
                    ax.set_ylabel("Accuracy")
                if relative_x:
                    ax.set_xlabel(f"Relative distance along path by {dkey}")
                else:
                    ax.set_xlabel(f"Distance along path by {dkey}")
        fig.legend()
        fig.show()
        return fig, axs

# def geodesic_projection_plane(geodesic_weights, plane):

#     # project all other points onto this plane
#     v1 = state_dict_to_numpy_array(geodesic_weights[0])
#     furthest_projected_points = [projection(v1, furthest_point_plane)]
#     for weights in geodesic_weights[1:]:
#         vi = state_dict_to_numpy_array(weights)
#         furthest_projected_points.append(projection(vi, furthest_point_plane))
    
#     return furthest_projected_points
    


# def plot_lmc_geodesic_comparison(
#     comparison_obj, # <-- as returned by compare_lmc_to_geodesic
#     path_types = ["geodesic", "lmc"],
#     quantity_sources = ["test", "train"],
#     distance_measure = "euclidean"
#     # ^ ... or a function from two models to distance, or "index"
# ):
#     fig, ax = plt.subplots()
#     linestyles = ["solid", "dotted"]
#     for i, path_type in enumerate(path_types):
#         path_results = comparison_obj[path_type]
#         xs = models_to_cumulative_distances(
#             path_results["models"],
#             distance_measure
#         )
#         for quantity_source in quantity_sources:
#             ax.plot(
#                 xs, path_results[quantity][quantity_source],
#                 linestyle=linestyles[i],
#                 label = f"{path_type} {quantity_source} {quantity}"
#             )
#     distance_name = distance_measure if isinstance(distance_measure, str) else "path"
#     ax.set_title(f"{', '.join(path_types)} {quantity} over {distance_name}")
#     ax.legend()
#     fig.show()
    

# def plot_losses_over_geodesic(
#     geodesic_path_models,
#     train_loader, test_loader, device,
#     n_points=25,
#     max_test_items=None,
#     **kwargs # see plot_lmc_geodesic_comparison for what these can be
# ):
#     comparison_obj = compare_lmc_to_geodesic(
#         geodesic_path_models,
#         train_loader, test_loader, device,
#         n_points,
#         max_test_items=max_test_items
#     )
#     plot_lmc_geodesic_comparison(comparison_obj, **kwargs)


# %%
if __name__ == "__main__":
    from models.mlp import MLP
    from utils.metrics import JSD_loss
    from utils.data import get_data_loaders
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from lmc import model_interpolation

    device, device_kwargs = get_device() # what are device_kwargs ???

    weights_a = torch.load("model_files/mlp_mnist_model_a.pt", map_location=torch.device('cpu'))
    weights_b = torch.load("model_files/mlp_mnist_model_b.pt", map_location=torch.device('cpu'))
    weights_bp = torch.load("model_files/mlp_mnist_model_pb.pt", map_location=torch.device('cpu'))

    trainset = datasets.MNIST(
        root="utils/data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    dl = DataLoader(trainset, 64)

    # batch = next(enumerate(dl))
    # idx, (batch_imgs, img_labels) = batch

    # %% (A cell for testing optimise_geodesic))

        # super_model,
        # dataloader,
        # lr=0.01,
        # num_epochs=1,
        # verbose=2,
        # loss_metric=JSD_loss
    smodel = SuperModel(MLP, 25, weights_a, weights_bp)
    # %%
    optimise_for_geodesic(
        smodel,
        loss_metric=JSD_loss,
        dataloader=dl,
        num_epochs=1,
        lr=1,
        verbose=1
    )
    # note that the above mutates smodel
    
    # %%
    comparison = compare_lmc_to_geodesic(
        smodel,
        MLP,
        dl,
        JSD_loss,
        verbose = 1
    )
    
    # %%

    plot_lmc_geodesic_comparison_obj(comparison)

    # NOW: SEE THE geodesic_experiments.ipynb NOTEBOOK FOR USE EXAMPLES

    
    # %%

    # === BELOW: pre-SuperModel example of use ===
    
    # plt.plot(geodesic_eval["accuracies"]["test"])
    # plt.plot(geodesic_eval["accuracies"]["train"])

    # comparison = compare_lmc_to_geodesic(
    #     geodesic_path_models,
    #     train_loader, test_loader, device,
    #     max_test_items = 512
    # )

    # # %% (Cell for testing plot_losses_over_geodesic)
    # model_a, model_b = MLP(), MLP()
    # load_checkpoint(model_a, "model_files/mlp_mnist_model_a.pt", device)
    # # model_a.load_state_dict(weights_a)
    # load_checkpoint(model_b, "model_files/mlp_mnist_model_pb.pt", device)
    # # model_b.load_state_dict(weights_bp)
    # device, device_kwargs = get_device()  # what are device_kwargs ???
    # model_a.to(device)
    # model_b.to(device)
    # train_loader, test_loader = get_data_loaders(
    #     dataset="mnist", train_kwargs={"batch_size": 4}, test_kwargs={"batch_size": 4}
    # )
    # device, device_kwargs = get_device()  # what are device_kwargs ???
    # plot_losses_over_geodesic(
    #     geodesic_path_models,
    #     train_loader, test_loader, device,
    #     max_test_items=512,
    #     distance_measure = "euclidean"
    # )

# %%
