# %%
import torch
import torch.nn.functional as f
import numpy as np
from utils.utils import state_dict_to_torch_tensor

def kl_div(logP, M):
    # kl_div over probability distributions
    # note: f.kl_div(logP, Q) = D_kl(Q||P) (the inputs are swapped), so we swap them below
    return f.kl_div(torch.log(M), logP.softmax(-1), reduction="none").sum(dim=-1).mean()

def JSD_loss(input_a, input_b):
    M = (input_a.softmax(-1)+input_b.softmax(-1)) / 2
    # note that f.kl_div takes log-probs
    return (kl_div(input_a, M) + kl_div(input_b, M)) / 2

def squared_euclid_dist(model_a, model_b, batch_imgs=None):
    if isinstance(model_a, dict):
        # make function work with pure state dicts too
        a_params = model_a
        b_params = model_b
    else:
        a_params = model_a.state_dict()
        b_params = model_b.state_dict()
    a_vect = state_dict_to_torch_tensor(a_params) 
    b_vect = state_dict_to_torch_tensor(b_params)
    return ((a_vect - b_vect)**2).sum()

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
        return total

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
