import torch
import copy
from typing import Callable
from tqdm import tqdm

from utils.data import get_data_loaders
from utils.utils import *
from utils.training_utils import test
from utils.weight_matching import *
from utils.plot import plot_interp_acc


def model_interpolation(model_a, model_b, train_loader, test_loader, device, n_points=25, verbose=2, max_test_items=None):
    "evaluates interpolation between two models of same architecture"

    # deepcopy to not change model outside function
    model_a_dict = copy.deepcopy(model_a.state_dict())
    model_b_dict = copy.deepcopy(model_b.state_dict())

    # points to interpolate on
    lambdas = torch.linspace(0, 1, steps=n_points)

    train_acc_list = []
    test_acc_list = []

    for i, lam in enumerate(lambdas):
        # linear interpolate model state dicts and load model
        lerp_model = lerp(lam, model_a_dict, model_b_dict)
        model_b.load_state_dict(lerp_model)

        # evaluate on train set
        train_loss, train_acc = test(
            model_b.to(device), device, train_loader, verbose=verbose, max_items=max_test_items
        )
        train_acc_list.append(train_acc)

        # evaluate on test set
        test_loss, test_acc = test(
            model_b.to(device), device, test_loader, verbose=verbose,
            max_items=max_test_items
        )
        test_acc_list.append(test_acc)

        if verbose >= 1:
            message = f"point {i+1}/{n_points}. "
            end = "\r"
            if verbose >= 2:
                message += f'lam = {lam}, train loss = {train_loss}, test loss = {test_loss}'
                end = "\n"
            print(message, end=end)

    return train_acc_list, test_acc_list


def permute_model(model_a, model_b, max_iter, verbose):

    permutation_spec = model_a.permutation_spec

    # finds permuation using weight matching
    permutation = weight_matching(permutation_spec, flatten_params(model_a), flatten_params(model_b), max_iter=max_iter, verbose=verbose)

    # applies permutation
    permuted_params = apply_permutation(permutation_spec, permutation, flatten_params(model_b))

    return permuted_params


def linear_mode_connect(
        model_factory: Callable,
        model_path_a: str,
        model_path_b: str,
        dataset,
        batch_size=4096,
        n_points=25,
        max_iter=20,
        verbose=2):
    """Given two models A and B, generates a permuted model B' and returns the metrics interpolated between A and B'.

    Args:
        model_factory (Callable): function called to generate a model. Could be a nn.Module class.
        model_path_a (str): Path where parameters of model A are stored
        model_path_b (str): Path where parameters of model B are stored
        dataset (str): Dataset which models were trained on
        batch_size (int, optional): Batch size for model inference. Defaults to 4098.
        n_points (int, optional): Number of interpolating points between A and B'. Defaults to 25.
        max_iter (int, optional): Max number of iterations for weight matching algorithm. Defaults to 20.
        verbose (int): Verbosity of logging. 0 is no outputs, 2 is highest level. Defaults to 2.

    Returns:
        permuted_params: dictionary of permuted parameters
        train_acc_naive: list of training accuracies during naive interpolation
        test_acc_naive: list of testing accuracies during naive interpolation
        train_acc_perm: list of training accuracies during weight-matched interpolation
        test_acc_perm list of testing accuracies during weight-matched interpolation
    """

    device, device_kwargs = get_device()

    # init models and load weights
    model_a = model_factory()
    load_checkpoint(model_a, model_path_a, device)

    model_b = model_factory()
    load_checkpoint(model_b, model_path_b, device)
    
    dataloader_kwargs = {'batch_size': batch_size}

    train_loader, test_loader = get_data_loaders(dataset, dataloader_kwargs, dataloader_kwargs, eval_only=True)

    if verbose >= 1:
        print('\nperforming naive interpolation')

    # interpolate naively between models
    
    train_acc_naive, test_acc_naive = model_interpolation(model_a, model_b, train_loader, test_loader, device, n_points=n_points, verbose=verbose)

    if verbose >= 1:
        print('\npermuting model')

    # perform weight matching and permute model b
    permuted_params = permute_model(model_a.cpu(), model_b.cpu(), max_iter, verbose)

    if verbose >= 1:
        print('\nperforming permuted interpolation')

    model_b.load_state_dict(permuted_params)

    # interpolate between model_a and permuted model_b
    train_acc_perm, test_acc_perm = model_interpolation(model_a, model_b, train_loader, test_loader, device, n_points=n_points, verbose=verbose)

    return permuted_params, train_acc_naive, test_acc_naive, train_acc_perm, test_acc_perm
