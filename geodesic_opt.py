# %%
from utils.metrics import JSD_loss
from utils.utils import lerp, get_device
# ^ THIS DOES WORK AAAAAAA :)
from random import randint
import torch

#%%


def metric_path_length(model_class, all_weights, loss_metric, data, device):
    length = 0
    for i in range(0, len(all_weights) - 1):
        model0, model1 = model_class(), model_class()
        model0.load_state_dict(all_weights[i])
        model1.load_state_dict(all_weights[i+1])
        model0.to(device)
        model1.to(device)
        length += loss_metric(model0, model1, data).detach().numpy()
    return length


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
        losses = []

        device, device_kwargs = get_device()

        while iterations < max_iterations and not CONVERGED:
            i = randint(1, n)

            model_before = model_class()
            model = model_class()
            model_after = model_class()

            model_before.load_state_dict(all_weights[i-1])
            model.load_state_dict(all_weights[i])
            model_after.load_state_dict(all_weights[i+1])

            model_before.to(device)
            model.to(device)
            model_after.to(device)

            opt = torch.optim.SGD(model.parameters(), lr=learning_rate)

            loss = (loss_metric(model_before, model, data) + loss_metric(model, model_after, data))

            opt.zero_grad()
            grad = loss.backward()
            opt.step()

            all_weights[i] = model.state_dict()
    
            iterations += 1

            losses.append(metric_path_length(model_class, all_weights, loss_metric, data, device))


            # ALSO: track distance moved
            # or change in L over entire path 
            # so we know if it's doing something

        return all_weights, losses
            

# %%
if __name__ == "__main__":
    from models.mlp import MLP
    from utils.metrics import JSD_loss
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    weights_a = torch.load("model_files/model_a.pt", map_location=torch.device('cpu'))
    weights_b = torch.load("model_files/model_b.pt", map_location=torch.device('cpu'))
    weights_bp = torch.load("model_files/permuted_model_b.pt", map_location=torch.device('cpu'))

    trainset = datasets.MNIST(
        root="utils/data", train=True, download=True,
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

    dl = DataLoader(trainset, 64)

    batch = next(enumerate(dl))
    idx, (batch_imgs, img_labels) = batch

    # %%

    opt_weights, losses = optimise_for_geodesic(
        MLP, weights_a, weights_bp,
        n = 10,
        loss_metric = JSD_loss,
        data = batch_imgs,
        max_iterations = 99,
        learning_rate = 0.1
    )

    plt.plot(losses)


# %%
