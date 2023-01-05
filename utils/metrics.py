# %%
import torch
import torch.nn.functional as f

def kl_div(logP, M):
    # kl_div over probability distributions
    return f.kl_div(logP.log_softmax(-1), M, reduction="none").sum(dim=-1).mean()

def KL_loss(model_a, model_b, batch_imgs):
    logP = model_a(batch_imgs)
    logQ = model_b(batch_imgs)
    return f.kl_div(logP.log_softmax(-1), logQ.log_softmax(-1), reduction="none", log_target=True)

def JSD_loss(model_a, model_b, batch_imgs):
    logP = model_a(batch_imgs)
    logQ = model_b(batch_imgs)
    # print(logP - logQ)
    M = (logP.softmax(-1)+logQ.softmax(-1)) / 2
    # note that f.kl_div takes log-probs
    return (kl_div(logP, M) + kl_div(logQ, M)) / 2


# TESTING BELOW:
# %%
import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath('.'))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from mode_connectivity.models.mlp import MLP
from data import get_data_loaders
import matplotlib.pyplot as plt

def load_model(model_class, path):
    model = model_class()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model
    

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
