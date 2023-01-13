# %%
import torch
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.metrics import JSD_loss, squared_euclid_dist, metric_path_length, models_to_cumulative_distances
from utils.objectives import heuristic_triplets, full_params
from utils.utils import lerp, get_device
from utils.training_utils import test
from utils.utils import load_checkpoint
from lmc import model_interpolation
# ^ THIS DOES WORK AAAAAAA :)


def optimise_for_geodesic(
    model_factory, weights_a, weights_b, n,
    loss_metric, objective_function, dataloader,
    max_iterations = 99, learning_rate = 0.01,
    return_losses = False,
    return_euclid_dist = False
):
    """
    Takes weights_a and weights_b of an instance of model_factory (an nn.Module
    that can be initialised without needing to pass in any args), a number of
    points n, a loss_metric (e.g. JSD_loss), a data loader,
    a maximum number of optimisation steps max_iterations to run for,
    a learning rate, and whether or not to return "losses", where in this case
    "loss" refers to the total length of the path from weights_a to weights_b
    when using loss_metric on model predictions at points between the path.

    Performs gradient descent with the weights at the intermediate points between
    model_a and model_b as the parameters, and loss_metric as the loss function.
    For performance 

    If return_losses=True, returns a tuple (list of weights, losses over training steps.)
    Otherwise, returns just the list of weights (a list where the first entry is weights_a,
    the last weights_b, and there are n-2 other weights between, where
    the weights are of the same form as returned by doing .state_dict() on a PyTorch model.)
    Note that calculating the losses at every optimisation step significantly slows down performance.
    """
    n -= 2
    # Change from convention where n is number of middle points,
    #  to convention where n is the number of points including the
    #  end points (as used by model_interpolation) 
    device, device_kwargs = get_device()
    
    data_iterator = iter(dataloader)

    all_weights = [
        lerp(i / (n + 1), weights_a, weights_b)
        for i in range(1, n + 1)
    ]
    all_weights = [weights_a] + all_weights + [weights_b]
    all_models = [model_factory() for i in range(0, n+2)]
    for i, model in enumerate(all_models):
        model.load_state_dict(all_weights[i])
        model.to(device)
    
    iterations = 0
    CONVERGED = False # TODO

    losses = [] if return_losses else None
    euclid_dists = [] if return_euclid_dist else None

    print("Optimising geodesic ...")
    for _ in tqdm(range(max_iterations)):
        opt, loss, batch_images, data_iterator = objective_function(
            all_models, loss_metric, data_iterator, dataloader,
            device, learning_rate, n
        )

        opt.zero_grad()
        grad = loss.backward()
        opt.step()

        all_weights[i] = model.state_dict()

        iterations += 1

        if return_losses:
            # note that this is a noisy measure,
            # because it only calculates the loss over
            # a single batch sample 
            losses.append(metric_path_length(all_models, loss_metric, batch_images))
        if return_euclid_dist:
            euclid_dists.append(metric_path_length(all_models, squared_euclid_dist, []))


        # ALSO: track distance moved
        # or change in L over entire path 
        # so we know if it's doing something
    output = [all_models]
    if return_losses:
        output.append(losses)
    if return_euclid_dist:
        output.append(euclid_dists)
    if len(output) > 1:
        return tuple(output)
    else:
        return output[0]

def evaluate_over_geodesic(
    geodesic_path_models,
    train_loader, test_loader, device,
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
    # model_a_weights = copy.deepcopy(model_a.state_dict())
    # model_b_weights = copy.deepcopy(model_b.state_dict())

    train_acc_list = []
    test_acc_list = []

    train_loss_list = []
    test_loss_list = []

    print("Calculating losses over geodesic:")
    for model in tqdm(geodesic_path_models):
        # evaluate on train set

        train_loss, train_acc = test(
            model.to(device), device, train_loader,
            verbose=0, max_items=max_test_items
        )
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

        # evaluate on test set
        test_loss, test_acc = test(
            model.to(device), device, test_loader,
            verbose=0, max_items=max_test_items
        )
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)
    
    # print(train_acc_list)
    # print(test_acc_list)

    return {
        "models": geodesic_path_models,
        "accuracies": {
            "train": train_acc_list,
            "test": test_acc_list
        },
        "losses": {
            "train": train_loss_list,
            "test": test_loss_list
        }
    }

def compare_lmc_to_geodesic(
    geodesic_path_models,
    train_loader, test_loader, device,
    verbose = True,
    max_test_items = None,
    model_factory = None
):
    n_points = len(geodesic_path_models)
    model_a = geodesic_path_models[0]
    model_b = geodesic_path_models[-1]
    if verbose:
        print("Calculating LMC train accuracies ...")
    lmc_results = model_interpolation(
        model_a, model_b,
        train_loader, test_loader,
        device, n_points, verbose=0,
        max_test_items=max_test_items,
        return_dict = True, model_factory = model_factory
    )

    if verbose:
        print("Calculating geodesic train accuracies ...")
    geodesic_results = evaluate_over_geodesic(
        geodesic_path_models,
        train_loader, test_loader, device,
        max_test_items=max_test_items
    )

    return {
        "lmc": lmc_results,
        "geodesic": geodesic_results
    }



def plot_lmc_geodesic_comparison(
    comparison_obj, # <-- as returned by compare_lmc_to_geodesic
    path_types = ["geodesic", "lmc"],
    quantity = "accuracies", # or "losses"
    quantity_sources = ["test", "train"],
    distance_measure = "euclidean"
    # ^ ... or a function from two models to distance, or "index"
):
    fig, ax = plt.subplots()
    linestyles = ["solid", "dotted"]
    for i, path_type in enumerate(path_types):
        path_results = comparison_obj[path_type]
        xs = models_to_cumulative_distances(
            path_results["models"],
            distance_measure
        )
        for quantity_source in quantity_sources:
            ax.plot(
                xs, path_results[quantity][quantity_source],
                linestyle=linestyles[i],
                label = f"{path_type} {quantity_source} {quantity}"
            )
    distance_name = distance_measure if isinstance(distance_measure, str) else "path"
    ax.set_title(f"{', '.join(path_types)} {quantity} over {distance_name}")
    ax.legend()
    fig.show()
    

def plot_losses_over_geodesic(
    geodesic_path_models,
    train_loader, test_loader, device,
    n_points=25,
    max_test_items=None,
    **kwargs # see plot_lmc_geodesic_comparison for what these can be
):
    comparison_obj = compare_lmc_to_geodesic(
        geodesic_path_models,
        train_loader, test_loader, device,
        n_points,
        max_test_items=max_test_items
    )
    plot_lmc_geodesic_comparison(comparison_obj, **kwargs)

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
        root="utils/data", train=True, download=True,
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

    dl = DataLoader(trainset, 64)

    # batch = next(enumerate(dl))
    # idx, (batch_imgs, img_labels) = batch

    # %% (A cell for testing optimise_geodesic))

    geodesic_path_models = optimise_for_geodesic(
        MLP, weights_a, weights_b, n=25,
        loss_metric=JSD_loss, dataloader=dl,
        objective_function=full_params,
        max_iterations=100,
        return_losses=False
    )
    
    # %% (A cell for testing model_geodesic_interpolation)
    device, device_kwargs = get_device() # what are device_kwargs ???
    model_a, model_b = MLP(), MLP()
    load_checkpoint(model_a, "model_files/mlp_mnist_model_a.pt", device)
    #model_a.load_state_dict(weights_a)
    load_checkpoint(model_b, "model_files/mlp_mnist_model_pb.pt", device)
    #model_b.load_state_dict(weights_bp)
    train_loader, test_loader = get_data_loaders(
        dataset="mnist", train_kwargs={"batch_size":4}, test_kwargs={"batch_size":4}
    )
    device, device_kwargs = get_device() # what are device_kwargs ???
    geodesic_eval = evaluate_over_geodesic(
        geodesic_path_models, train_loader, test_loader, device,
        max_test_items=1024
    )

    plt.plot(geodesic_eval["accuracies"]["test"])
    plt.plot(geodesic_eval["accuracies"]["train"])

    # %% (Cell for testing plot_losses_over_geodesic)
    model_a, model_b = MLP(), MLP()
    load_checkpoint(model_a, "model_files/mlp_mnist_model_a.pt", device)
    # model_a.load_state_dict(weights_a)
    load_checkpoint(model_b, "model_files/mlp_mnist_model_pb.pt", device)
    # model_b.load_state_dict(weights_bp)
    device, device_kwargs = get_device() # what are device_kwargs ???
    model_a.to(device)
    model_b.to(device)
    train_loader, test_loader = get_data_loaders(
        dataset="mnist", train_kwargs={"batch_size":4}, test_kwargs={"batch_size":4}
    )
    device, device_kwargs = get_device() # what are device_kwargs ???
    plot_losses_over_geodesic(
        geodesic_path_models,
        train_loader, test_loader, device,
        max_test_items=512,
        distance_measure = "euclidean"
    )

# %%
