# %%
import torch
import torch.nn.functional as F

def JSD_loss(logits_P, logits_Q):

    P = torch.softmax(logits_P)
    Q = torch.softmax(logits_Q)

    M = (P + Q) / 2

    logP, logQ, logM = torch.log(logP), torch.log(logQ), torch.log(logM)

    return (F.kl_div(logM, logP) + F.kl_div(logM, logQ)) / 2
