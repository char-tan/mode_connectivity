# %%
import torch
import torch.nn.functional as f
from utils.utils import state_dict_to_torch_tensor

def kl_div(logP, M):
    # kl_div over probability distributions
    # note: f.kl_div(logP, Q) = D_kl(Q||P) (the inputs are swapped), so we swap them below
    return f.kl_div(torch.log(M), logP.softmax(-1), reduction="none").sum(dim=-1).mean()

def KL_loss(model_a, model_b, batch_imgs):
    logP = model_a(batch_imgs)
    logQ = model_b(batch_imgs)
    return f.kl_div(logP.log_softmax(-1), logQ.log_softmax(-1), reduction="none", log_target=True)

def JSD_loss(model_a, model_b, batch_imgs):
    logP = model_a(batch_imgs)
    logQ = model_b(batch_imgs)
    M = (logP.softmax(-1)+logQ.softmax(-1)) / 2
    # note that f.kl_div takes log-probs
    return (kl_div(logP, M) + kl_div(logQ, M)) / 2

def squared_euclid_dist(model_a, model_b, batch_imgs):
    a_params = model_a.state_dict()
    b_params = model_b.state_dict()
    a_vect = state_dict_to_torch_tensor(a_params) 
    b_vect = state_dict_to_torch_tensor(b_params)
    return ((a_vect - b_vect)**2).sum()
        

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
