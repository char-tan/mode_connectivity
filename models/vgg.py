import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self):
        super().__init__()

        conv_cfg = [
                (64, 64), 
                (128, 128), 
                (256, 256, 256), 
                (512, 512, 512), 
                (512, 512, 512)]

        # TODO input and output dims of classifier depend on dataset
        classifier_cfg = {'input_dim' : 512, 'output_dim' : 10, 'width' : 4096}

        self.features = self._make_conv(conv_cfg)
        self.classifier = self._make_classifier(**classifier_cfg)

    def forward(self, x):

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def _make_conv(self, cfg):

        layers = []
        in_channels = 3

        # block is a tuple describing layers between pooling
        for block in cfg:

            # each element of block is a layer
            for out_channels in block:

                # add the conv layer to layers
                layers += [
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                        nn.GroupNorm(1, out_channels), # groupnorm with 1 group = layernorm
                        nn.ReLU(),
                        ]

                in_channels = out_channels

            # add the pooling layer at end of block
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)

    def _make_classifier(self, input_dim = 512, output_dim = 10, width = 4096):

        layers = []

        layers += [nn.Linear(input_dim, width), nn.ReLU()]
        layers += [nn.Linear(width, width), nn.ReLU()]
        layers += [nn.Linear(width, output_dim), nn.Softmax()]

        return nn.Sequential(*layers)

#def vgg16_permutation_spec() -> PermutationSpec:
#  layers_with_conv = [3,7,10,14,17,20,24,27,30,34,37,40]
#  layers_with_conv_b4 = [0,3,7,10,14,17,20,24,27,30,34,37]
#  layers_with_bn = [4,8,11,15,18,21,25,28,31,35,38,41]
#  dense = lambda name, p_in, p_out, bias = True: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out, )}
#  return permutation_spec_from_axes_to_perm({
#      # first features
#      "features.0.weight": ( "P_Conv_0",None, None, None),
#      "features.1.weight": ( "P_Conv_0", None),
#      "features.1.bias": ( "P_Conv_0", None),
#      "features.1.running_mean": ( "P_Conv_0", None),
#      "features.1.running_var": ( "P_Conv_0", None),
#      "features.1.num_batches_tracked": (),
#
#      **{f"features.{layers_with_conv[i]}.weight": ( f"P_Conv_{layers_with_conv[i]}", f"P_Conv_{layers_with_conv_b4[i]}", None, None, )
#        for i in range(len(layers_with_conv))},
#      **{f"features.{i}.bias": (f"P_Conv_{i}", )
#        for i in layers_with_conv + [0]},
#      # bn
#      **{f"features.{layers_with_bn[i]}.weight": ( f"P_Conv_{layers_with_conv[i]}", None)
#        for i in range(len(layers_with_bn))},
#      **{f"features.{layers_with_bn[i]}.bias": ( f"P_Conv_{layers_with_conv[i]}", None)
#        for i in range(len(layers_with_bn))},
#      **{f"features.{layers_with_bn[i]}.running_mean": ( f"P_Conv_{layers_with_conv[i]}", None)
#        for i in range(len(layers_with_bn))},
#      **{f"features.{layers_with_bn[i]}.running_var": ( f"P_Conv_{layers_with_conv[i]}", None)
#        for i in range(len(layers_with_bn))},
#      **{f"features.{layers_with_bn[i]}.num_batches_tracked": ()
#        for i in range(len(layers_with_bn))},
#
#      **dense("classifier", "P_Conv_40", "P_Dense_0", False),
#})
