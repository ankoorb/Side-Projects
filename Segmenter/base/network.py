import torch
import torch.nn as nn

from utils.helpers import _make_divisible


class ConvBnReLU(nn.Module):
    """
    [CONV]-[BN]-[ReLU6]
    """

    def __init__(self, inCh, outCh, stride):
        super(ConvBnReLU, self).__init__()
        self.inCh = inCh  # Number of input channels
        self.outCh = outCh  # Number of output channels
        self.stride = stride  # Stride
        self.conv = nn.Sequential(
            nn.Conv2d(self.inCh, self.outCh, 3, stride=self.stride, padding=1, bias=False),
            nn.BatchNorm2d(outCh),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class InvertedResidual(nn.Module):
    """
    [EXP:CONV_1x1-BN-ReLU6]-[DW:CONV_3x3-BN-ReLU6]-[PW:CONV_1x1-BN] with identity shortcut 
    and dilation.
    """

    def __init__(self, inCh, outCh, t, s, r):
        super(InvertedResidual, self).__init__()
        self.inCh = inCh
        self.outCh = outCh
        self.t = t  # t: expansion factor
        self.r = r  # r: dilation
        if self.r > 1:
            self.s = 1  # s: Stride
            self.padding = self.r  # Atrous Conv padding same as dilation rate
        else:
            self.s = s  # s: Stride
            self.padding = 1
        self.identity_shortcut = (self.inCh == self.outCh) and (self.s == 1)  # L:506 Keras official code

        # Bottleneck block
        self.block = nn.Sequential(
            # Expansition Conv
            nn.Conv2d(self.inCh, self.t * self.inCh, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.t * self.inCh),
            nn.ReLU6(inplace=True),

            # Depthwise Conv
            nn.Conv2d(self.t * self.inCh, self.t * self.inCh, kernel_size=3, stride=self.s, padding=self.padding,
                      dilation=self.r, groups=self.t * self.inCh, bias=False),
            nn.BatchNorm2d(self.t * self.inCh),
            nn.ReLU6(inplace=True),

            # Pointwise Linear Conv (Projection): i.e. No non-linearity
            nn.Conv2d(self.t * self.inCh, self.outCh, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.outCh),
        )

    def forward(self, x):
        if self.identity_shortcut:
            return x + self.block(x)
        else:
            return self.block(x)


class PointwiseConv(nn.Module):
    def __init__(self, inCh, outCh):
        super(PointwiseConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inCh, outCh, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(outCh),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# MobileNetV2
class MobileNetV2(nn.Module):
    """
    MobileNetV2 feature extractor modified to include dilation for DeepLabV3+. 
    NOTE: Last conv Layer and classification layer removed.
    """

    def __init__(self, params):
        super(MobileNetV2, self).__init__()
        self.params = params
        self.first_inCh = 3

        self.c = [_make_divisible(c * self.params.alpha, 8) for c in self.params.c]

        # Layer-0
        self.layer0 = nn.Sequential(ConvBnReLU(self.first_inCh, self.c[0], self.params.s[0]))

        # Layer-1
        self.layer1 = self._make_layer(self.c[0], self.c[1], self.params.t[1], self.params.s[1],
                                       self.params.n[1], self.params.r[1])

        # Layer-2: Image size: 512 -> [IRB-2] -> Output size: 128 (low level feature: 128 * 4 = 512)
        self.layer2 = self._make_layer(self.c[1], self.c[2], self.params.t[2], self.params.s[2],
                                       self.params.n[2], self.params.r[2])

        # Layer-3
        self.layer3 = self._make_layer(self.c[2], self.c[3], self.params.t[3], self.params.s[3],
                                       self.params.n[3], self.params.r[3])

        # Layer-4
        self.layer4 = self._make_layer(self.c[3], self.c[4], self.params.t[4], self.params.s[4],
                                       self.params.n[4], self.params.r[4])

        # Layer-5: Image size: 512 -> [IRB-5] -> Output size: 32, so output stride = 16 achieved
        self.layer5 = self._make_layer(self.c[4], self.c[5], self.params.t[5], self.params.s[5],
                                       self.params.n[5], self.params.r[5])

        # Layer-6: Apply dilation rate = 2
        self.layer6 = self._make_layer(self.c[5], self.c[6], self.params.t[6], self.params.s[6],
                                       self.params.n[6], self.params.r[6])

        # Layer-7: Apply dilation rate = 2
        self.layer7 = self._make_layer(self.c[6], self.c[7], self.params.t[7], self.params.s[7],
                                       self.params.n[7], self.params.r[7])

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, inCh, outCh, t, s, n, r):
        layers = []
        for i in range(n):
            # First layer of each sequence has a stride s and all others use stride 1
            if i == 0:
                layers.append(InvertedResidual(inCh, outCh, t, s, r))
            else:
                layers.append(InvertedResidual(inCh, outCh, t, 1, r))

            # Update input channel for next IRB layer in the block
            inCh = outCh
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        low_level_features = self.layer2(x)  # [512, 512]/4 = [128, 128]
        x = self.layer3(low_level_features)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return x, low_level_features

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def MobileNet(pretrained=True, **kwargs):
    """
    Constructs a MobileNet V2 model.

    Parameters
    ----------
    pretrained: bool, use ImageNet pretrained model or not.
    n_class: int, 1000 classes in ImageNet data.
    weight_file: str, path to pretrained weights
    """
    weight_file = kwargs.pop('weight_file', '')
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = torch.load(weight_file)
        model.load_state_dict(state_dict)
    return model

