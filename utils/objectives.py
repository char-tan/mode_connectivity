import torch
from random import randint
from itertools import chain
from geodesic_opt import metric_path_length


def heuristic_triplets(all_models, loss_metric, data_iterator, data_loader, device, learning_rate, n):
    """Objective Function """
    i = randint(1, n)

    model_before= all_models[i-1]
    model = all_models[i]
    model_after = all_models[i+1]

    opt = torch.optim.SGD(model.parameters(), lr=learning_rate)

    try: 
        batch_images, __ = next(data_iterator)
    except StopIteration:
        data_iterator = iter(data_loader)
        batch_images, __ = next(data_iterator)

    batch_images = batch_images.to(device)
    loss = (loss_metric(model_before, model, batch_images) + loss_metric(model, model_after, batch_images))

    return opt, loss, batch_images, data_iterator

def full_params(all_models, loss_metric, data_iterator, data_loader, device, learning_rate, n):
    """Objective function """
    opt = torch.optim.SGD(chain(*[model.parameters() for model in all_models]), lr=learning_rate)

    try: 
        batch_images, __ = next(data_iterator)
    except StopIteration:
        data_iterator = iter(data_loader)
        batch_images, __ = next(data_iterator)

    batch_images = batch_images.to(device)

    loss = metric_path_length(all_models, loss_metric, batch_images, track_grad = True)

    return opt, loss, batch_images, data_iterator

    
    