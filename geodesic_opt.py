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

import torch.nn.functional as F

def metric_path_length(outputs, loss_metric=JSD_loss):
    length = 0
    for i in range(0, len(outputs) - 1):
        length += loss_metric(outputs[i], outputs[i+1])
    return length

def acc_on_path(outputs, target, loss_function=F.cross_entropy):
  
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
    super_model, dataloader,
    lr = 0.01,
    num_epochs = 1
):

    device, _ = get_device()

    path_lengths = []
    sq_euc_dists = []
    path_accs = []

    optimizer = torch.optim.SGD(super_model.parameters(), lr=lr)

    print("Optimising geodesic ...")

    for epoch_idx in range(num_epochs):

        for batch_idx, (data, target) in enumerate(dataloader):

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = super_model(data)
      
            path_length = metric_path_length(outputs)

            sq_euc_dist = super_model.sq_euc_dist()

            path_acc = acc_on_path(outputs, target)

            mean_path_acc = path_acc.mean()

            print(f'batch {batch_idx} | path length {path_length} | sq euc dist {sq_euc_dist} | mean path acc {mean_path_acc}')
    
            path_length.backward()
            optimizer.step()

            path_lengths.append(path_length)
            sq_euc_dists.append(sq_euc_dist)
            path_accs.append(path_acc)

    return path_lengths, sq_euc_dists, path_accs

def losses_over_geodesic(
    model_factory, model_a, model_b, train_loader, test_loader, device, n_points=25,
    verbose=False,
    loss_metric=JSD_loss, max_iterations=99,
    max_test_items=1024
):
    """
    Takes models (often probably these will be models that have been
    LMC connected), data loaders, a device, a number of points, a loss
    metric.

    Optimises the geodesic path between the two models, based on loss_metric,
    and for a maximum max_iterations number of optimisation steps. (The
    optimisation is based on the TRAINING DATA.)

    Returns the test and train losses over the optimised geodesic path.
    These can be compared to e.g. the test and train losses over the
    un-optimised pure LMC path.
    """
    # deepcopy to not change model outside function
    model_a_weights = copy.deepcopy(model_a.state_dict())
    model_b_weights = copy.deepcopy(model_b.state_dict())

    geodesic_path_models = optimise_for_geodesic(
        model_factory, model_a_weights, model_b_weights, n_points,
        loss_metric, train_loader, max_iterations=max_iterations,
        return_losses=False
    )

    train_acc_list = []
    test_acc_list = []

    print("Calculating losses over geodesic:")
    for model in tqdm(geodesic_path_models):
        # evaluate on train set

        train_loss, train_acc = test(
            model.to(device), device, train_loader,
            verbose=0, max_items=max_test_items
        )
        train_acc_list.append(train_acc)

        # evaluate on test set
        test_loss, test_acc = test(
            model.to(device), device, test_loader,
            verbose=0, max_items=max_test_items
        )
        test_acc_list.append(test_acc)
    
    # print(train_acc_list)
    # print(test_acc_list)

    return train_acc_list, test_acc_list

def compare_losses_over_geodesic(
    model_factory, model_a, model_b,
    train_loader, test_loader, device, n_points=25,
    loss_metric=JSD_loss, max_iterations=99, verbose = True,
    max_test_items = None
):
    if verbose:
        print("Calculating LMC train accuracies ...")
    lmc_train_accs, lmc_test_accs = model_interpolation(
        model_a, model_b,
        train_loader, test_loader,
        device, n_points, verbose=0,
        max_test_items=max_test_items
    )

    if verbose:
        print("Calculating geodesic train accuracies ...")
    geodesic_train_accs, geodesic_test_accs = losses_over_geodesic(
        model_factory, model_a, model_b,
        train_loader, test_loader, device, n_points=n_points,
        loss_metric=loss_metric, max_iterations=max_iterations,
        verbose=verbose,
        max_test_items=max_test_items
    )

    return ((lmc_train_accs, lmc_test_accs), (geodesic_train_accs, geodesic_test_accs))

def plot_losses_over_geodesic(
    model_factory, model_a, model_b,
    train_loader, test_loader, device,
    n_points=25,
    loss_metric=JSD_loss,
    max_iterations=99,
    max_test_items=None
):
    ((lmc_train_accs, lmc_test_accs), (geodesic_train_accs, geodesic_test_accs)) = compare_losses_over_geodesic(
        model_factory, model_a, model_b,
        train_loader, test_loader, device,
        n_points,
        loss_metric,
        max_iterations,
        max_test_items=max_test_items
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

    weights_a = torch.load("model_files/model_a.pt", map_location=torch.device('cpu'))
    weights_b = torch.load("model_files/model_b.pt", map_location=torch.device('cpu'))
    weights_bp = torch.load("model_files/permuted_model_b.pt", map_location=torch.device('cpu'))

    trainset = datasets.MNIST(
        root="utils/data", train=True, download=True,
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

    dl = DataLoader(trainset, 64)

    # batch = next(enumerate(dl))
    # idx, (batch_imgs, img_labels) = batch

    # %% (A cell for testing optimise_geodesic))

    opt_models, losses = optimise_for_geodesic(
        MLP, weights_a, weights_b,
        n = 10,
        loss_metric = JSD_loss,
        dataloader = dl,
        max_iterations = 10, # <-- THIS IS VERY LOW
        learning_rate = 0.1,
        return_losses=True
    )

    plt.plot(losses)
    
    # %% (A cell for testing model_geodesic_interpolation)
    device, device_kwargs = get_device() # what are device_kwargs ???
    model_a, model_b = MLP(), MLP()
    load_checkpoint(model_a, "model_files/model_a.pt", device)
    #model_a.load_state_dict(weights_a)
    load_checkpoint(model_b, "model_files/permuted_model_b.pt", device)
    #model_b.load_state_dict(weights_bp)
    train_loader, test_loader = get_data_loaders(
        dataset="mnist", train_kwargs={"batch_size":4}, test_kwargs={"batch_size":4}
    )
    device, device_kwargs = get_device() # what are device_kwargs ???
    losses_along_path = losses_over_geodesic(
        MLP, model_a, model_b, train_loader, test_loader, device, n_points=25,
        loss_metric=JSD_loss, max_iterations=99,
        max_test_items=1024
    )

    plt.plot(losses_along_path[0])
    plt.plot(losses_along_path[1])

    # %% (Cell for testing plot_losses_over_geodesic)
    model_a, model_b = MLP(), MLP()
    load_checkpoint(model_a, "model_files/model_a.pt", device)
    # model_a.load_state_dict(weights_a)
    load_checkpoint(model_b, "model_files/permuted_model_b.pt", device)
    # model_b.load_state_dict(weights_bp)
    device, device_kwargs = get_device() # what are device_kwargs ???
    model_a.to(device)
    model_b.to(device)
    train_loader, test_loader = get_data_loaders(
        dataset="mnist", train_kwargs={"batch_size":4}, test_kwargs={"batch_size":4}
    )
    device, device_kwargs = get_device() # what are device_kwargs ???
    plot_losses_over_geodesic(
        MLP, model_a, model_b, train_loader, test_loader, device, n_points=25,
        loss_metric=JSD_loss, max_iterations=4,
        max_test_items=1024
    )

# %%
