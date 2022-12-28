import torch
import copy

from utils.data import get_data_loaders
from utils.utils import *
from utils.training_utils import test
from utils.weight_matching import *
from utils.plot import plot_interp_acc

def model_interpolation(model_a, model_b, train_loader, test_loader, device, n_points = 25):
    "evaluates interpolation between two models of same architecture"

    # deepcopy to not change model outside function
    model_a_dict = copy.deepcopy(model_a.state_dict())
    model_b_dict = copy.deepcopy(model_b.state_dict())

    # points to interpolate on
    lambdas = torch.linspace(0, 1, steps=n_points)

    train_acc_list = []
    test_acc_list = []

    for lam in lambdas:

        # linear interpolate model state dicts and load model
        lerp_model = lerp(lam, model_a_dict, model_b_dict)
        model_b.load_state_dict(lerp_model)

        # evaluate on train set
        train_loss, train_acc = test(model_b.to(device), device, train_loader, verbose=0)
        train_acc_list.append(train_acc)

        # evaluate on test set
        test_loss, test_acc = test(model_b.to(device), device, test_loader, verbose=0)
        test_acc_list.append(test_acc)

        print(f'lam = {lam}, train loss = {train_loss}, test loss = {test_loss}')

    return train_acc_list, test_acc_list

def permute_model(model_a, model_b, max_iter):

    permutation_spec = model_a.permutation_spec

    # finds permuation using weight matching
    permutation = weight_matching(permutation_spec, flatten_params(model_a), flatten_params(model_b), max_iter=max_iter)

    # applies permutation
    permuted_params = apply_permutation(permutation_spec, permutation, flatten_params(model_b))

    return permuted_params

def linear_mode_connect(
        model_factory,
        model_path_a,
        model_path_b,
        dataset,
        batch_size = 4098,
        n_points = 25,
        max_iter = 20):

    device, device_kwargs = get_device()

    # init models and load weights
    model_a = model_factory()
    load_checkpoint(model_a, model_path_a, device)

    model_b = model_factory()
    load_checkpoint(model_b, model_path_b, device)

    dataloader_kwargs = {'batch_size' : 512} # TODO can prob increase (no grads)

    train_loader, test_loader = get_data_loaders(dataset, dataloader_kwargs, dataloader_kwargs)

    print('\nperforming naive interpolation')

    # interpolate naively between models
    train_acc_naive, test_acc_naive = model_interpolation(model_a, model_b, train_loader, test_loader, device, n_points = n_points)

    print('\npermuting model')

    # perform weight matching and permute model b
    permuted_params = permute_model(model_a.cpu(), model_b.cpu(), max_iter)

    print('\nperforming permuted interpolation')

    model_b.load_state_dict(permuted_params)

    # interpolate between model_a and permuted model_b
    train_acc_perm, test_acc_perm = model_interpolation(model_a, model_b, train_loader, test_loader, device, n_points = n_points)

    fig = plot_interp_acc(n_points, train_acc_naive, test_acc_naive,
                    train_acc_perm, test_acc_perm)
