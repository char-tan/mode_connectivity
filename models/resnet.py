import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import numpy as np

# never saw anyting about specific initialisation

# def conv_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         init.xavier_uniform_(m.weight, gain=np.sqrt(2))
#         init.constant_(m.bias, 0)
#     elif classname.find('GroupNorm') != -1:
#         init.constant_(m.weight, 1)
#         init.constant_(m.bias, 0)


class Block(nn.Module):
    def __init__(self, in_chans, out_chans, stride, spatial_dim):
        super().__init__()

        # stride = stride
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.LayerNorm([out_chans, spatial_dim, spatial_dim])

        # stride = 1
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.LayerNorm([out_chans, spatial_dim, spatial_dim])

        # if stride == 2, need to halve spatial dims of input for skip connection
        if stride != 1:
            assert stride == 2

            # non-standard way of doing this, same as git-rebasin implementation
            self.shortcut = nn.Sequential(*[
                nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.LayerNorm([out_chans, spatial_dim, spatial_dim])
                ])

        else:
            self.shortcut = lambda x: x

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
        self.norm1 = nn.LayerNorm([16 * wm, 32, 32])

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

    def forward(self, x):

        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.block_groups(x)
        x = F.avg_pool2d(x, kernel_size=8, stride=8)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = F.softmax(x)

        return x

#def resnet20_permutation_spec() -> PermutationSpec:
#  conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None, )}
#  norm = lambda name, p: {f"{name}.weight": (p, ), f"{name}.bias": (p, )}
#  dense = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out, )}
#
#  # This is for easy blocks that use a residual connection, without any change in the number of channels.
#  easyblock = lambda name, p: {
#  **norm(f"{name}.bn1", p),
#  **conv(f"{name}.conv1", p, f"P_{name}_inner"),
#  **norm(f"{name}.bn2", f"P_{name}_inner"),
#  **conv(f"{name}.conv2", f"P_{name}_inner", p),
#  }
#
#  # This is for blocks that use a residual connection, but change the number of channels via a Conv.
#  shortcutblock = lambda name, p_in, p_out: {
#  **norm(f"{name}.bn1", p_in),
#  **conv(f"{name}.conv1", p_in, f"P_{name}_inner"),
#  **norm(f"{name}.bn2", f"P_{name}_inner"),
#  **conv(f"{name}.conv2", f"P_{name}_inner", p_out),
#  **conv(f"{name}.shortcut.0", p_in, p_out),
#  **norm(f"{name}.shortcut.1", p_out),
#  }
#
#  return permutation_spec_from_axes_to_perm({
#    **conv("conv1", None, "P_bg0"),
#    #
#    **shortcutblock("layer1.0", "P_bg0", "P_bg1"),
#    **easyblock("layer1.1", "P_bg1",),
#    **easyblock("layer1.2", "P_bg1"),
#    #**easyblock("layer1.3", "P_bg1"),
#
#    **shortcutblock("layer2.0", "P_bg1", "P_bg2"),
#    **easyblock("layer2.1", "P_bg2",),
#    **easyblock("layer2.2", "P_bg2"),
#    #**easyblock("layer2.3", "P_bg2"),
#
#    **shortcutblock("layer3.0", "P_bg2", "P_bg3"),
#    **easyblock("layer3.1", "P_bg3",),
#    **easyblock("layer3.2", "P_bg3"),
#   # **easyblock("layer3.3", "P_bg3"),
#
#    **norm("bn1", "P_bg3"),
#
#    **dense("linear", "P_bg3", None),
#
#})
