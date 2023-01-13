# %%
import torch
import torch.nn.functional as F

def JSD_loss(logits_P, logits_Q):

    P = torch.softmax(logits_P, dim=-1)
    Q = torch.softmax(logits_Q, dim=-1)

    M = (P + Q) / 2

    logP, logQ, logM = torch.log(P), torch.log(Q), torch.log(M)

    # PT KL Div is reversed input
    JSD = (F.kl_div(logM, logP, log_target=True, reduction='batchmean') + 
                  F.kl_div(logM, logQ, log_target=True, reduction='batchmean')) / 2
    
    return JSD