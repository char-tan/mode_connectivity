import functools
from functools import partial
import copy
import math

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from config import Config
from utils import (
    Timer,
    print_training_details,
    split_parameters,
    format_for_table,
    LOGGING_COLUMNS_LIST,
)
from data import prep_data, get_batches
from resnet import resnet20


def train_epoch(net, data, loss_fn, opt, lr_sched, epoch, config):

    net.train()

    train_loss, train_acc = [], []

    # precompute all batches
    batches = get_batches(data.train, config, train=True)

    for inputs, targets in batches:

        opt.zero_grad(set_to_none=True)

        outputs = net(inputs)

        loss = loss_fn(outputs, targets)

        train_acc.append((outputs.detach().argmax(-1) == targets).float().mean().item())
        train_loss.append(loss.detach().item())

        loss.backward()
        opt.step()

        lr_sched.step()

    return np.mean(train_loss), np.mean(train_acc)


@torch.no_grad()
def eval_epoch(net, data, loss_fn, epoch, config):

    net.eval()

    eval_loss, eval_acc = [], []

    # precompute all batches
    batches = get_batches(data.eval, config, train=False)

    for inputs, targets in batches:

        outputs = net(inputs)
        loss = loss_fn(outputs, targets)

        eval_acc.append((outputs.detach().argmax(-1) == targets).float().mean().item())
        eval_loss.append(loss.detach().item())

    return np.mean(eval_loss), np.mean(eval_acc)


def setup_training(config):

    net = resnet20(wm=config.wm).to(config.device).to(config.dtype)

    # split the params into non-bias and bias
    non_bias_params, bias_params = split_parameters(net)

    # two param groups
    opt = torch.optim.SGD(
        [
            {
                "params": non_bias_params,
                "lr": config.lr,
                "weight_decay": config.weight_decay,
            },
            {
                "params": bias_params,
                "lr": config.lr * config.bias_scaler,
                "weight_decay": config.weight_decay * config.bias_scaler,
            },
        ],
        momentum=config.momentum,
        nesterov=True,
    )

    steps_per_epoch = math.ceil(config.num_train_samples / config.batchsize)

    lr_sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=[config.lr, config.lr * config.bias_scaler],
        total_steps=steps_per_epoch * config.num_epochs,
        pct_start=config.pct_start,
        # cycle_momentum=False,
        div_factor=config.div_factor,
        final_div_factor=config.final_div_factor,
    )

    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    return net, opt, lr_sched, loss_fn


def train_experiment(data, config):

    net, opt, lr_sched, loss_fn = setup_training(config)

    print_training_details(LOGGING_COLUMNS_LIST, column_heads_only=True)

    timer = Timer(config)

    best_acc = None

    for epoch in range(config.num_epochs):

        timer.start()

        train_loss, train_acc = train_epoch(
            net, data, loss_fn, opt, lr_sched, epoch, config
        )

        eval_loss, eval_acc = eval_epoch(net, data, loss_fn, epoch, config)

        if best_acc is None:
            best_acc = eval_acc
        elif eval_acc > best_acc:
            print('saving')
            torch.save(net.state_dict(), config.model_path)
            best_acc = eval_acc

        epoch_time = timer.report()

        print_training_details(
            list(map(partial(format_for_table, locals=locals()), LOGGING_COLUMNS_LIST)),
            is_final_entry=(epoch == config.num_epochs - 1),
        )


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    use_cuda = torch.cuda.is_available()

    for wm in [1, 4, 16]:

        config = Config(
            model_path = None,
            dtype=torch.float16 if use_cuda else torch.float32,
            device="cuda" if use_cuda else "cpu",
            num_train_samples = 50000 if use_cuda else 128,
            wm=wm,
        )

        data = prep_data(config)

        for seed in [0, 1, 2]:

            print(seed, wm)

            torch.manual_seed(seed)
            config.model_path = f'model_wm{wm}_seed{seed}.pt'
            train_experiment(data, config)
