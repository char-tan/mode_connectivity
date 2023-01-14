import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from utils.weight_matching import permutation_spec_from_axes_to_perm
import sys
import numpy as np


class Block(nn.Module):
    def __init__(self, in_chans, out_chans, stride, spatial_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=out_chans)

        # stride = 1
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=out_chans)

        # if stride == 2, need to halve spatial dims of input for skip connection
        if stride != 1:
            assert stride == 2

            # non-standard way of doing this, same as git-rebasin implementation
            self.shortcut = nn.Sequential(*[
                nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=2, padding=0, bias=False),
                nn.GroupNorm(num_groups=1, num_channels=out_chans)
                ])

        else:
            self.shortcut = nn.Identity()

    def forward(self, x):

        # go through layers
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)

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

        self.conv1 = nn.Conv2d(3, 16 * wm, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=16*wm)

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

        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.block_groups(x)
        x = F.avg_pool2d(x, kernel_size=8, stride=8)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

    def _permutation_spec(self):

        conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None)}
        norm = lambda name, p: {f"{name}.weight": (p,), f"{name}.bias": (p,)}
        linear = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out,)}

        block = lambda name, p_in, p_inner: {
                **conv(f"{name}.conv1", p_in, p_inner),
                **norm(f"{name}.norm1", p_inner),
                **conv(f"{name}.conv2", p_inner, p_in),
                **norm(f"{name}.norm2", p_in),
        }

        shortcut_block = lambda name, p_in, p_inner, p_out: {
                **conv(f"{name}.conv1", p_in, p_inner),
                **norm(f"{name}.norm1", p_inner),
                **conv(f"{name}.conv2", p_inner, p_out),
                **norm(f"{name}.norm2", p_out),
                **conv(f"{name}.shortcut.0", p_in, p_out),
                **norm(f"{name}.shortcut.1", p_out),
        }

        perm_dict = {}

        # first conv + norm
        perm_dict.update({**conv("conv1", None, "P_BG_0"), **norm("norm1", "P_BG_0")})


        # block group 0
        perm_dict.update({**block(f"block_groups.0.blocks.0", f"P_BG_0", f"P_BG_0_IN_0")})
        perm_dict.update({**block(f"block_groups.0.blocks.1", f"P_BG_0", f"P_BG_0_IN_1")})
        perm_dict.update({**block(f"block_groups.0.blocks.2", f"P_BG_0", f"P_BG_0_IN_2")})

        # block group 1
        perm_dict.update({**shortcut_block(f"block_groups.1.blocks.0", f"P_BG_0", f"P_BG_1_IN_0", "P_BG_1")})
        perm_dict.update({**block(f"block_groups.1.blocks.1", f"P_BG_1", f"P_BG_1_IN_1")})
        perm_dict.update({**block(f"block_groups.1.blocks.2", f"P_BG_1", f"P_BG_1_IN_2")})

        # block group 2
        perm_dict.update({**shortcut_block(f"block_groups.2.blocks.0", f"P_BG_1", f"P_BG_2_IN_0", "P_BG_2")})
        perm_dict.update({**block(f"block_groups.2.blocks.1", f"P_BG_2", f"P_BG_2_IN_1")})
        perm_dict.update({**block(f"block_groups.2.blocks.2", f"P_BG_2", f"P_BG_2_IN_2")})
        perm_dict.update({**linear("linear", "P_BG_2", None)})

        return permutation_spec_from_axes_to_perm(perm_dict)

