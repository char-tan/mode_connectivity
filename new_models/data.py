import os
import math
from functools import partial
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

import torchvision
from torchvision import transforms


CIFAR10_MEAN = torch.tensor(
    [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
)
CIFAR10_STD = torch.tensor([0.24703225141799082, 0.24348516474564, 0.26158783926049628])


@dataclass
class TensorDataSubset:
    images: torch.Tensor
    targets: torch.Tensor


@dataclass
class TensorData:
    train: TensorDataSubset
    eval: TensorDataSubset


def data_to_tensor(dataset: torch.utils.data.Dataset, config):

    # load data onto gpu
    images = torch.tensor(dataset.data, device=config.device)
    targets = torch.tensor(dataset.targets, device=config.device)

    # BHWC -> BCHW and [0, 255] -> [0, 1]
    images = images.permute(0, 3, 1, 2) / 255

    return TensorDataSubset(images, targets)


def make_tensor_dataset(
    train_data: torch.utils.data.Dataset, eval_data: torch.utils.data.Dataset, config
):
    return TensorData(
        *[data_to_tensor(subset, config) for subset in [train_data, eval_data]]
    )


def normalize_images(images: torch.Tensor, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    return images - mean.view(1, -1, 1, 1).to(images.device) / std.view(1, -1, 1, 1).to(
        images.device
    )


def prep_data(config):

    # get dataset
    cifar10 = torchvision.datasets.CIFAR10("cifar10/", download=True, train=True)
    cifar10_eval = torchvision.datasets.CIFAR10("cifar10/", download=False, train=False)

    # reformat images, move to GPU
    data = make_tensor_dataset(cifar10, cifar10_eval, config)

    # so I can debug on cpu without my machine dying
    if config.num_train_samples < 50000:
        data.train.images = data.train.images[:config.num_train_samples]
        data.train.targets = data.train.targets[:config.num_train_samples]
        data.eval.images = data.eval.images[:config.num_train_samples]
        data.eval.targets = data.eval.targets[:config.num_train_samples]

    # normalize dataset, cast to fp16
    data.train.images = normalize_images(data.train.images).to(config.dtype)
    data.eval.images = normalize_images(data.eval.images).to(config.dtype)

    # pad the training images
    if config.pad_amount:
        data.train.images = F.pad(
            data.train.images, (config.pad_amount,) * 4, "reflect"
        )

    return data


# TODO TODO TODO need to check
def make_random_square_masks(inputs, mask_size):
    """
    This is actually (I believe) a pretty clean implementation of how to do something like this, since shifted-square masks unique to each depth-channel can actually be rather
    tricky in practice. That said, if there's a better way, please do feel free to submit it! This can be one of the harder parts of the code to understand (though I personally get
    stuck on the fold/unfold process for the lower-level convolution calculations.
    """

    is_even = int(mask_size % 2 == 0)
    in_shape = inputs.shape

    # seed centers of squares to cutout boxes from, in one dimension each
    mask_center_y = torch.empty(
        in_shape[0], dtype=torch.long, device=inputs.device
    ).random_(mask_size // 2 - is_even, in_shape[-2] - mask_size // 2 - is_even)
    mask_center_x = torch.empty(
        in_shape[0], dtype=torch.long, device=inputs.device
    ).random_(mask_size // 2 - is_even, in_shape[-1] - mask_size // 2 - is_even)

    # measure distance, using the center as a reference point
    to_mask_y_dists = torch.arange(in_shape[-2], device=inputs.device).view(
        1, 1, in_shape[-2], 1
    ) - mask_center_y.view(-1, 1, 1, 1)
    to_mask_x_dists = torch.arange(in_shape[-1], device=inputs.device).view(
        1, 1, 1, in_shape[-1]
    ) - mask_center_x.view(-1, 1, 1, 1)

    to_mask_y = (to_mask_y_dists >= (-(mask_size // 2) + is_even)) * (
        to_mask_y_dists <= mask_size // 2
    )
    to_mask_x = (to_mask_x_dists >= (-(mask_size // 2) + is_even)) * (
        to_mask_x_dists <= mask_size // 2
    )

    final_mask = (
        to_mask_y * to_mask_x
    )  ## Turn (y by 1) and (x by 1) boolean masks into (y by x) masks through multiplication. Their intersection is square, hurray! :D

    return final_mask


@torch.no_grad()
def batch_cutout(inputs, patch_size):

    cutout_batch_mask = make_random_square_masks(inputs, patch_size)

    cutout_batch = torch.where(cutout_batch_mask, torch.zeros_like(inputs), inputs)

    return cutout_batch


@torch.no_grad()
def batch_crop(inputs, crop_size):

    crop_mask_batch = make_random_square_masks(inputs, crop_size)

    cropped_batch = torch.masked_select(inputs, crop_mask_batch).view(
        inputs.shape[0], inputs.shape[1], crop_size, crop_size
    )

    return cropped_batch


@torch.no_grad()
def batch_flip_lr(batch_images, flip_chance=0.5):
    return torch.where(
        torch.rand_like(batch_images[:, 0, 0, 0].view(-1, 1, 1, 1)) < flip_chance,
        torch.flip(batch_images, (-1,)),
        batch_images,
    )


@torch.no_grad()
def get_batches(data: TensorData, config, train=False):

    images = data.images
    targets = data.targets

    num_samples = targets.shape[0]

    permutation = torch.randperm(num_samples, device=config.device)

    # apply transormations to train only
    if train:
        if config.pad_amount:
            images = batch_crop(images, config.crop_size)

        images = batch_flip_lr(images)

        # if non-zero do cutouts
        if config.cutout_size:
            images = batch_cutout(images, patch_size=config.cutout_size)

    # convert memory format
    images = images.to(memory_format=torch.channels_last)

    num_batches = math.ceil((num_samples) / config.batchsize)

    for idx in range(num_batches):

        slice_start = idx * config.batchsize
        slice_end = min((idx + 1) * config.batchsize, num_samples)

        yield (
            images.index_select(0, permutation[slice_start:slice_end]),
            targets.index_select(0, permutation[slice_start:slice_end]),
        )
