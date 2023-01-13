# %%
import torch
import torch.nn.functional as f
from utils.utils import state_dict_to_torch_tensor

def kl_div(logP, M):
    # kl_div over probability distributions
    # note: f.kl_div(logP, Q) = D_kl(Q||P) (the inputs are swapped), so we swap them below
    return f.kl_div(torch.log(M), logP.softmax(-1), reduction="none").sum(dim=-1).mean()

def JSD_loss(input_a, input_b):
    M = (input_a.softmax(-1)+input_b.softmax(-1)) / 2
    # note that f.kl_div takes log-probs
    return (kl_div(input_a, M) + kl_div(input_b, M)) / 2

def squared_euclid_dist(model_a, model_b, batch_imgs):
    a_params = model_a.state_dict()
    b_params = model_b.state_dict()
    a_vect = state_dict_to_torch_tensor(a_params) 
    b_vect = state_dict_to_torch_tensor(b_params)
    return ((a_vect - b_vect)**2).sum()

