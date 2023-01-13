# %%
import torch
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from evaluate import evaluate_supermodel, evaluate_lmc
from super import SuperModel

from utils.metrics import JSD_loss, squared_euclid_dist, metric_path_length, models_to_cumulative_distances, acc_on_path
from utils.objectives import heuristic_triplets, full_params
from utils.utils import lerp, get_device
from utils.training_utils import test
from utils.utils import load_checkpoint, intervals_to_cumulative_sums
from utils.metrics import JSD_loss
from lmc import evaluate_lmc, model_interpolation

# ^ THIS DOES WORK AAAAAAA :)



def optimise_for_geodesic(
    super_model,
    dataloader,
    lr=0.01,
    num_epochs=1,
    verbose=1,
    loss_metric=JSD_loss
):

    """
    pass in a super_model and dataloader, it will optimise this model for geodesic and return batch-wise path length and accuracy
    """

    device, _ = get_device()

    path_lengths = []
    sq_euc_dists = []

    optimizer = torch.optim.SGD(super_model.parameters(), lr=lr)

    print("Optimising geodesic ...")
    # for _ in tqdm(range(max_iterations)):
    #     opt, loss, batch_images, data_iterator = objective_function(
    #         all_models, loss_metric, data_iterator, dataloader,
    #         device, learning_rate, n
    #     )

    for epoch_idx in range(num_epochs):
        if verbose > 0:
            print(f"Epoch {epoch_idx} of {num_epochs}")
        for batch_idx, (data, target) in tqdm(list(enumerate(dataloader))):

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = super_model(data)

            path_length = metric_path_length(outputs, loss_metric=loss_metric)

            sq_euc_dist = super_model.sq_euc_dist()

            if verbose >= 2:
                print(
                    f"batch {batch_idx} | path length {path_length} | sq euc dist {sq_euc_dist}"
                )

            path_length.backward()
            optimizer.step()

            path_lengths.append(path_length.item())
            sq_euc_dists.append(sq_euc_dist.item())

        if verbose == 1:
            print(
                f"epoch {epoch_idx} | path length {np.mean(path_lengths)} | sq euc dist {np.mean(sq_euc_dists)}"
            )

    return path_lengths, sq_euc_dists

def compare_lmc_to_geodesic(
    geodesic_smodel,
    model_factory,
    dataloader,
    distance_metric,
    verbose = 1,
):
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
        distance_metric=distance_metric,
        verbose=1
    )

    if verbose:
        print("Calculating geodesic train accuracies ...")
    geodesic_path_lengths, geodesic_path_accs = evaluate_supermodel(
        geodesic_smodel,
        dataloader,
        distance_metric,
        verbose
    )

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


def plot_lmc_geodesic_comparison_obj(comparison_obj):
    fig, ax = plt.subplots()
    lmc_xs = intervals_to_cumulative_sums(comparison_obj["lmc"]["lengths"])
    lmc_accs = comparison_obj["lmc"]["accuracies"]
    geodesic_xs = intervals_to_cumulative_sums(comparison_obj["geodesic"]["lengths"])
    geodesic_accs = comparison_obj["geodesic"]["accuracies"]
    ax.plot(lmc_xs, lmc_accs, label="lmc")
    ax.plot(geodesic_xs, geodesic_accs, label="geodesic")
    ax.legend()
    fig.show()

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
