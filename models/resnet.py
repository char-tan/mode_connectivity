import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import numpy as np

from utils.weight_matching import permutation_spec_from_axes_to_perm


class Block(nn.Module):
    def __init__(self, in_chans, out_chans, stride, spatial_dim):
        super().__init__()

        # stride = stride
        self.conv0 = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm0 = nn.LayerNorm([out_chans, spatial_dim, spatial_dim])

        # stride = 1
        self.conv1 = nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.LayerNorm([out_chans, spatial_dim, spatial_dim])

        # if stride == 2, need to halve spatial dims of input for skip connection
        if stride != 1:
            assert stride == 2

            # non-standard way of doing this, same as git-rebasin implementation
            self.shortcut = nn.Sequential(*[
                nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=2, padding=0, bias=False),
                nn.LayerNorm([out_chans, spatial_dim, spatial_dim])
                ])

        else:
            self.shortcut = lambda x: x

    def forward(self, x):

        # go through layers
        y = x
        y = self.conv0(y)
        y = self.norm0(y)
        y = F.relu(y)
        y = self.conv1(y)
        y = self.norm1(y)

        # go through shortcut
        x = self.shortcut(x)

        return F.relu(y + x)


class BlockGroup(nn.Module):
    def __init__(self, in_chans, out_chans, blocks, stride, spatial_dim):
        super().__init__()

        self.blocks = nn.Sequential(
            *[Block(in_chans=in_chans, out_chans=out_chans, stride=stride, spatial_dim=spatial_dim)] +
            [Block(in_chans=out_chans, out_chans=out_chans, stride=1, spatial_dim=spatial_dim) for _ in range(blocks - 1)]
            )

    def forward(self, x):
        return self.blocks(x)


class ResNet(nn.Module):
    def __init__(self, width_multiplier=1, num_classes=10):
        super().__init__()

        wm = width_multiplier

        self.conv = nn.Conv2d(3, 16 * wm, kernel_size=3, padding=1, bias=False)
        self.norm = nn.LayerNorm([16 * wm, 32, 32])

        group_chans = [16 * wm, 32 * wm, 64 * wm]
        group_blocks = [3, 3, 3]  # for resnet20
        group_strides = [1, 2, 2]
        group_spatial_dims = [32, 16, 8]

        block_groups = []

        in_chans = 16 * wm

        for c, b, s, sd in zip(group_chans, group_blocks, group_strides, group_spatial_dims):
            block_groups.append(BlockGroup(in_chans=in_chans, out_chans=c, blocks=b, stride=s, spatial_dim=sd))
            in_chans = c

        self.block_groups = nn.Sequential(*block_groups)

        self.linear = nn.Linear(group_chans[-1], num_classes)

        self.permutation_spec = self._permutation_spec()

    def forward(self, x):

        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.block_groups(x)
        x = F.avg_pool2d(x, kernel_size=8, stride=8)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = F.softmax(x)

        return x

    def _permutation_spec(self):

        conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None)}
        norm = lambda name, p: {f"{name}.weight": (p,), f"{name}.bias": (p,)}
        linear = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out,)}

        block = lambda name, p_in, p_inner, p_out: {
                **conv(f"{name}.conv0", p_in, p_inner),
                **norm(f"{name}.norm0", p_inner),
                **conv(f"{name}.conv1", p_inner, p_out),
                **norm(f"{name}.norm1", p_out),
        }

        shortcut = lambda name, p_in, p_out: {
                **conv(f"{name}.shortcut.0", p_in, p_out),
                **norm(f"{name}.shortcut.1", p_out),
        }

        perm_dict = {}

        # first conv + norm
        perm_dict.update({**conv("conv", None, "P_0.0.0"), **norm("norm", "P_0.0.0")})

        # blockgroups
        for i in range(3):
            for j in range(3):

                block_name = f"block_groups.{i}.blocks.{j}"
                p_in = f"P_{i}.{j}.0"
                p_inner = f"P_{i}.{j}.1"

                # needs to match following block / blockgroup
                p_out = f"P_{i + (j + 1) // 3}.{(j + 1) % 3}.0"

                perm_dict.update({**block(block_name, p_in, p_inner, p_out)})

                # add shortcut
                if j == 0 and i > 0:
                    perm_dict.update({**shortcut(block_name, p_in, p_out)})

        # linear
        perm_dict.update({**linear("linear", "P_3.0.0", None)})

        return permutation_spec_from_axes_to_perm(perm_dict)
