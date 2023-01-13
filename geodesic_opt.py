# %%
import torch
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.utils import lerp, get_device
from utils.training_utils import test
from utils.utils import load_checkpoint
from utils.metrics import JSD_loss
from lmc import model_interpolation

# ^ THIS DOES WORK AAAAAAA :)

import numpy as np

import torch.nn.functional as F


def metric_path_length(outputs, loss_metric=JSD_loss, return_stepwise=False):
    """
    computes distribution space path length using loss_metric by default returns total but can return np.array of step-wise
    """

    total = 0
    stepwise = []

    for i in range(0, len(outputs) - 1):

        length = loss_metric(outputs[i], outputs[i + 1])

        total += length
        stepwise.append(length.item())

    if return_stepwise:
        return np.array(stepwise)
    else:
        return length


def acc_on_path(outputs, target, loss_function=F.cross_entropy):
    """
    outputs is the output of super_model (list of model outputs) computes loss wrt target for each list element
    """

    accs = []

    # I DO THIS PER BATCH NOT CUMULATIVE AS IN TRAIN / TEST
    for output in outputs:
        pred = output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item()
        accs.append(correct / target.shape[0] * 100)

    return torch.tensor(accs)


def optimise_for_geodesic(
    super_model,
    dataloader,
    lr=0.01,
    num_epochs=1,
    verbose=2,
):

    """
    pass in a super_model and dataloader, it will optimise this model for geodesic and return batch-wise path length and accuracy
    """

    device, _ = get_device()

    path_lengths = []
    sq_euc_dists = []

    optimizer = torch.optim.SGD(super_model.parameters(), lr=lr)

    print("Optimising geodesic ...")

    for epoch_idx in range(num_epochs):

        for batch_idx, (data, target) in enumerate(dataloader):

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = super_model(data)

            path_length = metric_path_length(outputs)

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


def evaluate_geodesic(super_model, dataloader, verbose=2):
    """
    evaluates super_model (a path of models) wrt to dataloader, returns mean step-wise path length, and mean model-wise accuracy
    """

    device, _ = get_device()

    path_accs = []
    path_lengths = []

    print("Calculating path length and acc over geodesic:")
    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(dataloader):

            data, target = data.to(device), target.to(device)

            outputs = super_model(data)

            path_length = metric_path_length(outputs, return_stepwise=True)

            path_acc = acc_on_path(outputs, target)

            mean_path_acc = path_acc.mean()

            if verbose >= 2:
                print(
                    f"batch {batch_idx} | path length {path_length.sum()} | mean path acc {mean_path_acc}"
                )

            path_accs.append(path_acc.cpu().numpy())
            path_lengths.append(path_length)

    return np.mean(np.stack(path_lengths), axis=0), np.mean(np.stack(path_accs), axis=0)


def compare_losses_over_geodesic(
    model_factory,
    model_a,
    model_b,
    train_loader,
    test_loader,
    device,
    n_points=25,
    loss_metric=JSD_loss,
    max_iterations=99,
    verbose=True,
    max_test_items=None,
):
    if verbose:
        print("Calculating LMC train accuracies ...")
    lmc_train_accs, lmc_test_accs = model_interpolation(
        model_a,
        model_b,
        train_loader,
        test_loader,
        device,
        n_points,
        verbose=0,
        max_test_items=max_test_items,
    )

    if verbose:
        print("Calculating geodesic train accuracies ...")
    geodesic_train_accs, geodesic_test_accs = losses_over_geodesic(
        model_factory,
        model_a,
        model_b,
        train_loader,
        test_loader,
        device,
        n_points=n_points,
        loss_metric=loss_metric,
        max_iterations=max_iterations,
        verbose=verbose,
        max_test_items=max_test_items,
    )

    return ((lmc_train_accs, lmc_test_accs), (geodesic_train_accs, geodesic_test_accs))


def plot_losses_over_geodesic(
    model_factory,
    model_a,
    model_b,
    train_loader,
    test_loader,
    device,
    n_points=25,
    loss_metric=JSD_loss,
    max_iterations=99,
    max_test_items=None,
):
    (
        (lmc_train_accs, lmc_test_accs),
        (geodesic_train_accs, geodesic_test_accs),
    ) = compare_losses_over_geodesic(
        model_factory,
        model_a,
        model_b,
        train_loader,
        test_loader,
        device,
        n_points,
        loss_metric,
        max_iterations,
        max_test_items=max_test_items,
    )
    fig, ax = plt.subplots()
    ax.plot(lmc_train_accs, label="LMC train acc.")
    ax.plot(lmc_test_accs, label="LMC test acc.")
    ax.plot(geodesic_train_accs, label="Geodesic train acc.")
    ax.plot(geodesic_test_accs, label="Geodesic test acc.")
    ax.legend()
    fig.show()


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

    weights_a = torch.load("model_files/model_a.pt", map_location=torch.device("cpu"))
    weights_b = torch.load("model_files/model_b.pt", map_location=torch.device("cpu"))
    weights_bp = torch.load(
        "model_files/permuted_model_b.pt", map_location=torch.device("cpu")
    )

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

    opt_models, losses = optimise_for_geodesic(
        MLP,
        weights_a,
        weights_b,
        n=10,
        loss_metric=JSD_loss,
        dataloader=dl,
        max_iterations=10,  # <-- THIS IS VERY LOW
        learning_rate=0.1,
        return_losses=True,
    )

    plt.plot(losses)

    # %% (A cell for testing model_geodesic_interpolation)
    device, device_kwargs = get_device()  # what are device_kwargs ???
    model_a, model_b = MLP(), MLP()
    load_checkpoint(model_a, "model_files/model_a.pt", device)
    # model_a.load_state_dict(weights_a)
    load_checkpoint(model_b, "model_files/permuted_model_b.pt", device)
    # model_b.load_state_dict(weights_bp)
    train_loader, test_loader = get_data_loaders(
        dataset="mnist", train_kwargs={"batch_size": 4}, test_kwargs={"batch_size": 4}
    )
    device, device_kwargs = get_device()  # what are device_kwargs ???
    losses_along_path = losses_over_geodesic(
        MLP,
        model_a,
        model_b,
        train_loader,
        test_loader,
        device,
        n_points=25,
        loss_metric=JSD_loss,
        max_iterations=99,
        max_test_items=1024,
    )

    plt.plot(losses_along_path[0])
    plt.plot(losses_along_path[1])

    # %% (Cell for testing plot_losses_over_geodesic)
    model_a, model_b = MLP(), MLP()
    load_checkpoint(model_a, "model_files/model_a.pt", device)
    # model_a.load_state_dict(weights_a)
    load_checkpoint(model_b, "model_files/permuted_model_b.pt", device)
    # model_b.load_state_dict(weights_bp)
    device, device_kwargs = get_device()  # what are device_kwargs ???
    model_a.to(device)
    model_b.to(device)
    train_loader, test_loader = get_data_loaders(
        dataset="mnist", train_kwargs={"batch_size": 4}, test_kwargs={"batch_size": 4}
    )
    device, device_kwargs = get_device()  # what are device_kwargs ???
    plot_losses_over_geodesic(
        MLP,
        model_a,
        model_b,
        train_loader,
        test_loader,
        device,
        n_points=25,
        loss_metric=JSD_loss,
        max_iterations=4,
        max_test_items=1024,
    )

# %%
