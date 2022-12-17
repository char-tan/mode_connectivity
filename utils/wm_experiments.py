import torch
import copy

from .data import get_data_loaders
from .utils import *
from .training import test
from .experiments import get_device

def model_interpolation(model_a, model_b, train_loader, test_loader, device, n_points = 25):
    "evaluates interpolation between two models of same architecture"

    # deepcopy to not change model outside function
    model_a_dict = copy.deepcopy(model_a.state_dict())
    model_b_dict = copy.deepcopy(model_b.state_dict())

    # points to interpolate on
    lambdas = torch.linspace(0, 1, steps=n_points)

    train_acc = []
    test_acc = []

    for lam in lambdas:

        # linear interpolate model state dicts and load model
        lerp_model = lerp(lam, model_a_dict, model_b_dict)
        model_b.load_state_dict(lerp_model)

        # evaluate on train set
        train_loss, train_acc = test(model_b.to(device), device, train_loader)
        train_acc.append(train_acc)

        print(train_loss)
        breakpoint()

        # evaluate on test set
        test_loss, test_acc = test(model_b.to(device), device, test_loader)
        test_acc.append(test_acc)

    return train_acc, test_acc

def permute_model(model_a, model_b, num_hidden_layers = 3):

    # produces specification for permuation?
    permutation_spec = mlp_permutation_spec(num_hidden_layers = num_hidden_layers) # TODO variable layers

    # finds permuation using weight matching
    permutation = weight_matching(permutation_spec, flatten_params(model_a), flatten_params(model_b))

    # applies permutation
    permuted_params = apply_permutation(permutation_spec, permutation, flatten_params(model_b))

    return permuted_params
def run_wm_experiment(
        model_factory,
        model_path_a,
        model_path_b,
        dataset,
        batch_size = 4098,
        n_points = 25):

    device, device_kwargs = get_device()

    # init models and load weights
    model_a = model_factory()
    #load_checkpoint(model_a, model_path_a)

    model_b = model_factory()
    #load_checkpoint(model_b, model_path_b)

    train_loader, test_loader = get_data_loaders(dataset, batch_size)

    # interpolate naively between models
    train_acc_naive, test_acc_naive = model_interpolation(model_a, model_b, train_loader, test_loader, device)

    # perform weight matching and permute model b
    permuted_params = permute_model(model_a, model_b, num_hidden_layers = 3)

    # interpolate between model_a and permuted model_b
    train_acc_perm, test_acc_perm = model_interpolation(model_a, permuted_params, train_loader, test_loader, device)

    fig = plot_interp_acc(n_points, train_acc_naive, test_acc_naive,
                    train_acc_clever, test_acc_clever)