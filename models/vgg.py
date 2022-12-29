import torch.nn as nn

from ..utils.weight_matching import permutation_spec_from_axes_to_perm


class VGG(nn.Module):
    def __init__(self):
        super().__init__()

        conv_cfg = [
            (64, 64),
            (128, 128),
            (256, 256, 256),
            (512, 512, 512),
            (512, 512, 512),
        ]

        classifier_cfg = {"input_dim": 512, "output_dim": 10, "width": 4096}

        self.features = self._make_conv(conv_cfg)
        self.classifier = self._make_classifier(**classifier_cfg)

        self.permutation_spec = self._permutation_spec()

    def forward(self, x):

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def _make_conv(self, cfg):

        layers = []
        in_channels = 3
        spatial_dim = 32

        # block is a tuple describing layers between pooling
        for block in cfg:

            # each element of block is a layer
            for out_channels in block:

                # add the conv layer to layers
                layers += [
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size=3, padding=1, bias=False
                    ),
                    nn.BatchNorm2d(out_channels),
                    #nn.LayerNorm([out_channels, spatial_dim, spatial_dim]),
                    nn.ReLU(),
                ]

                in_channels = out_channels

            # add the pooling layer at end of block
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            spatial_dim = spatial_dim // 2

        return nn.Sequential(*layers)

    def _make_classifier(self, input_dim=512, output_dim=10, width=4096):

        layers = []

        layers += [nn.Linear(input_dim, width), nn.ReLU()]
        layers += [nn.Linear(width, width), nn.ReLU()]
        layers += [nn.Linear(width, output_dim), nn.Softmax()]

        return nn.Sequential(*layers)

    def _permutation_spec(self):

        # layer descriptions
        is_conv = [3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
        follows_conv = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37]
        is_norm = [4, 8, 11, 15, 18, 21, 25, 28, 31, 35, 38, 41]

        return permutation_spec_from_axes_to_perm(
            {
                # first conv
                "features.0.weight": ("P_Conv_0", None, None, None),
                # first norm
                "features.1.weight": ("P_Conv_0", None),
                "features.1.bias": ("P_Conv_0", None),
                # conv layers
                **{
                    f"features.{is_conv[i]}.weight": (
                        f"P_Conv_{is_conv[i]}",
                        f"P_Conv_{follows_conv[i]}",
                        None,
                        None,
                    )
                    for i in range(len(is_conv))
                },
                **{f"features.{i}.bias": (f"P_Conv_{i}",) for i in is_conv + [0]},
                # layer norms
                **{
                    f"features.{is_norm[i]}.weight": (f"P_Conv_{is_conv[i]}", None)
                    for i in range(len(is_norm))
                },
                **{
                    f"features.{is_norm[i]}.bias": (f"P_Conv_{is_conv[i]}", None)
                    for i in range(len(is_norm))
                },
                # classifier
                "classifier.0.weight": ("P_Dense_0", f"P_Conv_{is_conv[-1]}"),
                "classifier.2.weight": (f"P_Dense_2", f"P_Dense_0"),
                "classifier.2.bias": ("P_Dense_2",),
                "classifier.4.weight": (None, "P_Dense_4"),
                "classifier.4.bias": (None,),
            }
        )
