import torch.nn as nn

from utils.weight_matching import permutation_spec_from_axes_to_perm

class MLP(nn.Module):
    def __init__(self, input=28*28):
        super().__init__()

        self.input = input
        self.layer0 = nn.Linear(input, 512)
        self.layer1 = nn.Linear(512, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 10)

        num_hidden_layers = 3

        """We assume that one permutation cannot appear in two axes of the same weight array."""
        self.permutation_spec = permutation_spec_from_axes_to_perm({
            "layer0.weight": ("P_0", None),
            **{f"layer{i}.weight": ( f"P_{i}", f"P_{i-1}") for i in range(1, num_hidden_layers)},
            **{f"layer{i}.bias": (f"P_{i}", ) for i in range(num_hidden_layers)},
            f"layer{num_hidden_layers}.weight": (None, f"P_{num_hidden_layers-1}"),
            f"layer{num_hidden_layers}.bias": (None, ),
            })

    def forward(self, x):
        x = x.view(-1, self.input)
        x = nn.functional.relu(self.layer0(x))
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = self.layer3(x)

        return nn.functional.log_softmax(x)
