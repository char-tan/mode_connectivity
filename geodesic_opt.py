# %%
from utils.metrics import JSD_loss
from utils.utils import lerp, load_model
# ^ THIS DOES WORK AAAAAAA :)
from random import randint
import torch

#%%
def optimise_for_geodesic(
    model_class, weights_a, weights_b, n, loss_metric, data,
    max_iterations = 99, learning_rate = 0.01
    ):
        all_weights = [
            lerp(i / (n + 1), weights_a, weights_b)
            for i in range(1, n + 1)
        ]
        # ^ weights we are optimising, i.e. not first or last
        all_weights = [weights_a] + all_weights + [weights_b]
        # NB: theta_1 != theta_a, theta_n != theta_b
        iterations = 0
        CONVERGED = False # TODO
        while iterations < max_iterations and not CONVERGED:
            i = randint(1, n)

            model_before = model_class()
            model = model_class()
            model_after = model_class()

            model_before.load_state_dict(all_weights[i-1])
            model.load_state_dict(all_weights[i])
            model_after.load_state_dict(all_weights[i+1])

            opt = torch.optim.SGD(model.parameters(), lr=learning_rate)

            loss = (loss_metric(model_before, model, data) + loss_metric(model, model_after, data))

            opt.zero_grad()
            grad = loss.backward()
            opt.step()

            all_weights[i] = model.state_dict()
    
            iterations += 1

            # ALSO: track distance moved
            # or change in L over entire path 
            # so we know if it's doing something

        return all_weights
            

# %%
