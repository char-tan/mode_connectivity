from lmc import linear_mode_connect
from models.mlp import MLP
from models.vgg import VGG
from models.resnet import ResNet

model = ResNet()

print(model)

for name, p in model.named_parameters():
    print(name)

ps = model.permutation_spec

ax = ps.axes_to_perm
perm = ps.perm_to_axes

print()

for i, j in ax.items():
    print(i, j)

print()

for i, j in perm.items():
    print(i, j)
