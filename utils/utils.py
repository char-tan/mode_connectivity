# %%
import sys
sys.path.insert(0, "..")

import copy
import numpy as np
import torch
from tqdm import tqdm
from utils.training_utils import test
from dataclasses import dataclass

@dataclass
class TwoDimensionalPlane:
  """Def ines a two dimensional plane in an n-dimensional space."""
  b1: np.array
  b2: np.array
  scale: float
  origin_vector: np.array


def load_checkpoint(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
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
        device_kwargs = {"num_workers": 1, "pin_memory": True}
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

def reconstruct(vector, example_state_dict):
  """Reconstruct a torch state_dict object from a numpy array representing a vector of model weights."""
  i = 0
  output = dict()
  for key, parameter in example_state_dict.items():
    shape_now = parameter.shape
    size_now = np.prod(shape_now)
    data_now = vector[i:i+size_now].reshape(shape_now)
    output[key] = torch.Tensor(data_now)
    i = i + size_now
  return output


def state_dict_to_numpy_array(model_params):
  """Flatten the state_dict of a model into a numpy array (vector) of weights."""
  keys = model_params.keys()
  v = np.concatenate([model_params[key].reshape([-1]).cpu() for key in keys],axis=0)
  return v

def generate_orthogonal_basis(v1, v2, v3):
  """Generate a two-dimensional plane, given any three points in space which define it."""
  basis1 = v2-v1
  basis1_normed = basis1 / np.sqrt(np.sum(basis1**2.0))
  basis2 = v3 - v1
  basis2 = basis2 - np.sum(basis2*basis1_normed)*basis1_normed #orthogonalization
  basis2_normed = basis2 / np.sqrt(np.sum(basis2**2.0))

  scale = np.sqrt(np.sum(basis1**2))
  origin_vector = v1

  return TwoDimensionalPlane(b1=basis1_normed, b2=basis2_normed, scale=scale, origin_vector=origin_vector)

def generate_loss_landscape_contour(model, device, train_loader, test_loader, plane:TwoDimensionalPlane, granularity:int=20):
  """Given a plane in the loss landscape generate the loss and acc at grid points in the plane."""
  t1s = np.linspace(-0.5,1.5,granularity+1)
  t2s = np.linspace(-0.5,1.5,granularity)

  test_acc_grid = np.zeros((len(t1s),len(t2s)))
  test_loss_grid = np.zeros((len(t1s),len(t2s)))
  train_acc_grid = np.zeros((len(t1s),len(t2s)))
  train_loss_grid = np.zeros((len(t1s),len(t2s)))

  example_state_dict = model.state_dict()
  for i1,t1 in tqdm(enumerate(t1s)):
    for i2,t2 in enumerate(t2s):

      new_flat_v = plane.origin_vector + plane.b1*t1*plane.scale + plane.b2*t2*plane.scale
      reconstructed_flat = reconstruct(new_flat_v, example_state_dict)
      model.load_state_dict(reconstructed_flat)

      model.eval()
      with torch.no_grad():
        test_loss, test_acc = test(model.to(device), device, test_loader, verbose=0)
        train_loss, train_acc = test(model.to(device), device, train_loader, verbose=0)
      
      test_acc_grid[i1,i2] = test_acc
      test_loss_grid[i1,i2] = test_loss
      train_acc_grid[i1,i2] = train_acc
      train_loss_grid[i1,i2] = train_loss

  return t1s, t2s, test_acc_grid, test_loss_grid, train_acc_grid, train_loss_grid

def projection(vector, plane:TwoDimensionalPlane):
  x = np.sum((vector - plane.origin_vector)*plane.b1)/plane.scale
  y = np.sum((vector - plane.origin_vector)*plane.b2)/plane.scale
  return x,y
# %%
