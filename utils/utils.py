import torch
import copy

def load_checkpoint(model, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)   

def get_device():
    use_cuda = torch.cuda.is_available()
    try:
        use_mps = torch.backends.mps.is_available()
    except AttributeError:
        use_mps = False
    if use_cuda:
        device = "cuda"
    elif use_mps:
        device = "mps"
    else:
        device = "cpu"
    device = torch.device(device)
    if use_cuda:
        device_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    else:
        device_kwargs = {}

    return device, device_kwargs

def flatten_params(model):
  return model.state_dict()

def lerp(lam, t1, t2):
  t3 = copy.deepcopy(t2)
  for p in t1: 
    t3[p] = (1 - lam) * t1[p] + lam * t2[p]
  return t3
