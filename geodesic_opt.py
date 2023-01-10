# %%
from random import randint
import torch
import copy

from utils.metrics import JSD_loss
from utils.utils import lerp, get_device
from utils.training_utils import test
# ^ THIS DOES WORK AAAAAAA :)

def metric_path_length(model_factory, all_weights, loss_metric, data, device):
    # data is a single batch
    length = 0
    for i in range(0, len(all_weights) - 1):
        model0, model1 = model_factory(), model_factory()
        model0.load_state_dict(all_weights[i])
        model1.load_state_dict(all_weights[i+1])
        model0.to(device)
        model1.to(device)
        if torch.cuda.is_available():
            length += loss_metric(model0, model1, data).detach().cpu().numpy()
        else:
            length += loss_metric(model0, model1, data).detach().numpy()
    return length


def optimise_for_geodesic(
    model_factory, weights_a, weights_b, n, loss_metric, dataloader,
    max_iterations = 99, learning_rate = 0.01,
    return_losses = False
):
    """
    Takes weights_a and weights_b of an instance of model_factory (an nn.Module
    that can be initialised without needing to pass in any args), a number of
    middle points n, a loss_metric (e.g. JSD_loss), a data loader,
    a maximum number of optimisation steps max_iterations to run for,
    a learning rate, and whether or not to return "losses", where in this case
    "loss" refers to the total length of the path from weights_a to weights_b
    when using loss_metric on model predictions at points between the path.

    Performs gradient descent with the weights at the intermediate points between
    model_a and model_b as the parameters, and loss_metric as the loss function.
    For performance 

    If return_losses=True, returns a tuple (list of weights, losses over training steps.)
    Otherwise, returns just the list of weights (a list where the first entry is weights_a,
    the last weights_b, and there are n other weights between (i.e. total n+2), where
    the weights are of the same form as returned by doing .state_dict() on a PyTorch model.)
    Note that calculating the losses at every optimisation step significantly slows down performance.
    """
    all_weights = [
        lerp(i / (n + 1), weights_a, weights_b)
        for i in range(1, n + 1)
    ]
    # ^ weights we are optimising, i.e. not first or last
    all_weights = [weights_a] + all_weights + [weights_b]
    # NB: theta_1 != theta_a, theta_n != theta_b
    iterations = 0
    CONVERGED = False # TODO
    losses = []

    device, device_kwargs = get_device()

    while iterations < max_iterations and not CONVERGED:
        i = randint(1, n)

        model_before = model_factory()
        model = model_factory()
        model_after = model_factory()

        model_before.load_state_dict(all_weights[i-1])
        model.load_state_dict(all_weights[i])
        model_after.load_state_dict(all_weights[i+1])

        model_before.to(device)
        model.to(device)
        model_after.to(device)

        opt = torch.optim.SGD(model.parameters(), lr=learning_rate)

        batch_images, batch_labels = next(iter(dataloader))
        batch_images = batch_images.to(device)
        loss = (loss_metric(model_before, model, batch_images) + loss_metric(model, model_after, batch_images))

        opt.zero_grad()
        grad = loss.backward()
        opt.step()

        all_weights[i] = model.state_dict()

        iterations += 1

        if return_losses:
            losses.append(metric_path_length(model_factory, all_weights, loss_metric, batch_images, device))


        # ALSO: track distance moved
        # or change in L over entire path 
        # so we know if it's doing something
    
    if return_losses:
        return all_weights, losses
    return all_weights

def losses_over_geodesic(
    model_factory, model_a, model_b, train_loader, test_loader, device, n_points=25,
    loss_metric=JSD_loss, max_iterations=99
):
    """
    Takes models (often probably these will be models that have been LMC connected),
    data loaders, a device, a number of points, a loss metric.

    Optimises the geodesic path between the two models, based on loss_metric, and for
    a maximum max_iterations number of optimisation steps. (The optimisation is based
    on the TRAINING DATA.)

    Returns the test and train losses over the optimised geodesic path. These can be compared
    to e.g. the test and train losses over the un-optimised pure LMC path.
    """
    # deepcopy to not change model outside function
    model_a_weights = copy.deepcopy(model_a.state_dict())
    model_b_weights = copy.deepcopy(model_b.state_dict())

    geodesic_path_model_weights = optimise_for_geodesic(
        model_factory, model_a_weights, model_b_weights, n_points,
        loss_metric, train_loader, max_iterations=max_iterations,
        return_losses=False
    )

    train_acc_list = []
    test_acc_list = []

    for i, weights in enumerate(geodesic_path_model_weights):
        model = model_factory()
        model.load_state_dict(weights)

        # evaluate on train set

        train_loss, train_acc = test(model.to(device), device, train_loader, verbose=0)
        train_acc_list.append(train_acc)

        # evaluate on test set
        test_loss, test_acc = test(model.to(device), device, test_loader, verbose=0)
        test_acc_list.append(test_acc)

    return train_acc_list, test_acc_list            

# %%
if __name__ == "__main__":
    from models.mlp import MLP
    from utils.metrics import JSD_loss
    from utils.data import get_data_loaders
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

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

    opt_weights, losses = optimise_for_geodesic(
        MLP, weights_a, weights_bp,
        n = 10,
        loss_metric = JSD_loss,
        dataloader = dl,
        max_iterations = 99,
        learning_rate = 0.1,
        return_losses=True
    )

    plt.plot(losses)
    
    # %% (A cell for testing model_geodesic_interpolation)
    model_a, model_b = MLP(), MLP()
    model_a.load_state_dict(weights_a)
    model_b.load_state_dict(weights_bp)
    train_loader, test_loader = get_data_loaders(
        dataset="mnist", train_kwargs={"batch_size":512}, test_kwargs={"batch_size":512}
    )
    device, device_kwargs = get_device() # what are device_kwargs ???
    losses_along_path = losses_over_geodesic(
        MLP, model_a, model_b, train_loader, test_loader, device, n_points=25,
        loss_metric=JSD_loss, max_iterations=99
    )

# %%
