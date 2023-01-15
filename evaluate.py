import copy
import numpy as np
import torch
from tqdm import tqdm
from super import SuperModel
from utils.metrics import JSD_loss, metric_path_length, squared_euclid_dist
import torch.nn.functional as f
from utils.utils import get_device

def models_to_cumulative_distances(models, distance_measure):
    if distance_measure == "index":
        return list(range(len(models)))
    elif distance_measure == "euclidean":
        distance_fn = squared_euclid_dist
    else:
        distance_fn = distance_measure
    distances = [0.0] + [
        distance_fn(models[i], models[i+1])
        for i in range(0, len(models) - 1)
    ]
    return torch.cumsum(torch.tensor(distances), dim=0)

def acc_on_path(outputs, target, loss_function=f.cross_entropy):
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

def evaluate_supermodel(
    super_model,
    dataloader,
    distance_metric=JSD_loss,
    verbose=1
):
    """
    evaluates super_model (a path of models) wrt to dataloader, returns mean step-wise path length, and mean model-wise accuracy
    """

    device, _ = get_device()

    path_accs = []
    path_lengths = []
    print("Calculating path length and acc over SuperModel:")
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(list(enumerate(dataloader))):
            data, target = data.to(device), target.to(device)
            outputs = super_model(data)
            path_length = metric_path_length(
                outputs,
                loss_metric=distance_metric,
                return_stepwise=True
            )
            path_acc = acc_on_path(outputs, target)
            mean_path_acc = path_acc.mean()
            if verbose >= 2:
                print(
                    f"batch {batch_idx} | path length {path_length.sum()} | mean path acc {mean_path_acc}"
                )
            path_accs.append(path_acc.cpu().numpy())
            path_lengths.append(path_length)
    return np.mean(np.stack(path_lengths), axis=0), np.mean(np.stack(path_accs), axis=0)

def evaluate_lmc(
    model_factory,
    n,
    weights_a,
    weights_b,
    dataloader,
    distance_metric=JSD_loss,
    verbose=1
):
    """
    evaluates the lmc path from model_a to model_b containing n points
    total (including model_a and model_b as the endpoints) with respect
    to dataloader, returns mean step-wise path length,
    and mean model-wise accuracy
    """
    device, _ = get_device()
    weights_a = copy.deepcopy(weights_a)
    weights_b = copy.deepcopy(weights_b)
    lmc_super_model = SuperModel(model_factory, n, weights_a, weights_b).to(device)
    return evaluate_supermodel(
        lmc_super_model,
        dataloader,
        distance_metric,
        verbose
    )