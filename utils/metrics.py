# %%
import torch
import torch.nn.functional as F
import numpy as np
from utils.utils import state_dict_to_torch_tensor


def JSD_loss(logits_P, logits_Q):

    P = torch.softmax(logits_P, dim=-1)
    Q = torch.softmax(logits_Q, dim=-1)

    M = (P + Q) / 2

    logP, logQ, logM = torch.log(P), torch.log(Q), torch.log(M)

    # PT KL Div is reversed input
    JSD = (F.kl_div(logM, logP, log_target=True, reduction='batchmean') +
                  F.kl_div(logM, logQ, log_target=True, reduction='batchmean')) / 2

    return JSD

def sqrt_JSD_loss(logits_P, logits_Q):
    return torch.sqrt(JSD_loss(logits_P, logits_Q))

def squared_euclid_dist(model_a, model_b, batch_imgs=None):
    if isinstance(model_a, torch.Tensor):
        a_vect = model_a
        b_vect = model_b
    elif isinstance(model_a, dict):
        # make function work with pure state dicts too
        a_vect = state_dict_to_torch_tensor(model_a)
        b_vect = state_dict_to_torch_tensor(model_b)
    else:
        a_vect = state_dict_to_torch_tensor(model_a.state_dict())
        b_vect = state_dict_to_torch_tensor(model_b.state_dict())
    return ((a_vect - b_vect)**2).sum()

def euclid_dist(model_a, model_b, batch_imgs=None):
    return torch.sqrt(squared_euclid_dist(model_a, model_b))

def index_distance(model_a, model_b, data=None):
    return torch.tensor(1)

def metric_path_length(outputs, loss_metric=JSD_loss, return_stepwise=False):
    """
    computes distribution space path length using loss_metric by default returns total but can return np.array of step-wise
    """

    total = 0
    sqrt_total = 0
    stepwise = []

    for i in range(0, len(outputs) - 1):

        length = loss_metric(outputs[i], outputs[i + 1])

        total += length
        sqrt_total += torch.sqrt(length)
        stepwise.append(length.item())

    if return_stepwise:
        return np.array(stepwise)
    else:
        return total, sqrt_total

# OLD VERSION:
# def metric_path_length(all_models, loss_metric, data, track_grad = False):
#     # data is a single batch
#     length = 0
#     n = len(all_models)

#     if track_grad:
#         length = torch.sum(torch.stack([loss_metric(all_models[i], all_models[i+1], data) for i in range(n-1)]))
#     else:
#         length = sum([loss_metric(all_models[i], all_models[i+1], data).detach() for i in range(n-1)]).detach()
#     # for i in range(0, len(all_models) - 1):
#     #     model0, model1 = all_models[i], all_models[i+1]
#     #     # length += loss_metric(model0, model1, data).detach().cpu().numpy()
#     #     length += loss_metric(model0, model1, data).detach()
#     return length


# def fisher_info_matrix(model, data):
#     # more or less pseudo code - have yet to check the correctness of this implementation
#     logP = model(data).log_softmax(-1)

#     fim = []
#     for x in range(logP.shape[0]):
#         for y in range(logP.shape[1]):
#             logP[x,y].backward()
#             param_grads_xy = torch.cat([
#                 param.grad.flatten()
#                 for name, param in model_a.named_parameters()
#             ])
#             grad_matrix_xy = torch.mm(torch.reshape(param_grads_xy, [param_grads_xy.shape[0], 1]), ), torch.reshape(param_grads_xy, [1, param_grads_xy.shape[0]])
#             fim += grad_matrix_xy * logP.softmax(-1)[x,y] / logP.shape[0]
#     return fim



# TESTING BELOW:
# %%
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    import os
    from models.mlp import MLP
    from utils.data import get_data_loaders
    import matplotlib.pyplot as plt

    def load_model(model_class, path):
        model = model_class()
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        return model

    print(os.getcwd())

    model_a = load_model(MLP, '../model_files/model_a.pt')
    model_b = load_model(MLP, '../model_files/model_b.pt')
    model_bp = load_model(MLP, '../model_files/permuted_model_b.pt')



    # %%
    #from mode_connectivity.utils.utils import get_device
    #device, device_kwargs = get_device()
    # device, device_kwargs = "cpu", {}
    # train_kwargs = {"batch_size": 13, "shuffle": True}
    # test_kwargs = {"batch_size": 13, "shuffle": False}
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    trainset = datasets.MNIST(
        root="./data", train=True, download=True,
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

    dl = DataLoader(trainset, 64)

    batch = next(enumerate(dl))
    idx, (batch_imgs, img_labels) = batch

# %%
