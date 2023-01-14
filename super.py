import torch
import torch.nn as nn

from utils.utils import state_dict_to_torch_tensor, lerp

class SuperModel(nn.Module):
    def __init__(self, model_factory, n, state_dict_a, state_dict_b):
        super().__init__()

        n -= 2
        # Change from convention where n is number of middle points,
        #  to convention where n is the number of points including the
        #  end points (as used by model_interpolation) 

        lerp_models = []

        for i in range(1, n+1):
            model = model_factory()
            weights = lerp(i / (n + 1), state_dict_a, state_dict_b)
            model.load_state_dict(weights)

            lerp_models.append(model)

        model_a = model_factory()
        model_a.load_state_dict(state_dict_a)
        model_a.requires_grad_(requires_grad=False)

        model_b = model_factory()
        model_b.load_state_dict(state_dict_b)
        model_b.requires_grad_(requires_grad=False)

        all_models = [model_a] + lerp_models + [model_b]

        self.models = nn.ModuleList(all_models)

    def forward(self, x):

        x = x.view(x.shape[0], -1)

        outputs = []

        for model in self.models:
            outputs.append(model(x.clone()))

        return outputs


    def sq_euc_dist(self):

        total = 0

        state_dict_a = self.models[0].state_dict()

        model_vec_a = state_dict_to_torch_tensor(state_dict_a)

        for model in self.models[1:]:
            
            state_dict_b = model.state_dict()

            model_vec_b = state_dict_to_torch_tensor(state_dict_b)

            total += ((model_vec_a - model_vec_b)**2).sum()

            model_vec_a = model_vec_b

        return total
