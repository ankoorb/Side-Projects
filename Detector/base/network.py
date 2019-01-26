import math
import torch
import torch.nn as nn


## References
# 1. https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v2.py
# 2. https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
# 3. https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py

def _make_divisible(v, divisor, min_value=None):
    """
    This function makes sure that number of channels number is divisible by 8.
    Source: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


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
    [CONV_1x1-BN-ReLU6]-[CONV_3x3-BN-ReLU6]-[CONV_1x1-BN] with identity shortcut.
    """

    def __init__(self, inCh, outCh, t, s):
        super(InvertedResidual, self).__init__()
        self.inCh = inCh
        self.outCh = outCh
        self.t = t  # t: expansion factor
        self.s = s  # s: Stride
        self.identity_shortcut = (self.inCh == self.outCh) and (self.s == 1)  # L:506 Keras official code

        # Bottleneck block
        self.block = nn.Sequential(
            # Expansition Conv
            nn.Conv2d(self.inCh, self.t * self.inCh, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.t * self.inCh),
            nn.ReLU6(inplace=True),

            # Depthwise Conv
            nn.Conv2d(self.t * self.inCh, self.t * self.inCh, kernel_size=3, stride=self.s, padding=1,
                      groups=self.t * self.inCh, bias=False),
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
    MobileNetV2 feature extractor for YOLOv3. NOTE: YOLOv3 uses convolutional layers only!

    Input: 416 x 416 x 3
    Last layer Pointwise conv output:13 x 13 x 1024 -> Large object detection
    5th layer Pointwise conv output: :26 x 26 x 512 -> Medium object detection
    3rd layer Pointwise conv output: 52 x 52 x 256 -> Small object detection
    """

    def __init__(self, params):
        super(MobileNetV2, self).__init__()
        self.params = params
        self.first_inCh = 3
        last_outCh = 1280

        self.c = [_make_divisible(c * self.params.alpha, 8) for c in self.params.c]
        # Last convolution has 1280 output channels for alpha <= 1
        self.last_outCh = _make_divisible(int(last_outCh * self.params.alpha),
                                          8) if self.params.alpha > 1.0 else last_outCh

        # NOTE: YOLOv3 makes predictions at 3 different scales: (1) In the last feature map layer: 13 x 13
        # (2) The feature map from 2 layers previous and upsample it by 2x: 26 x 26
        # (3) The feature map from 2 layers previous and upsample it by 2x: 52 x 52

        # Layer-0
        self.layer0 = nn.Sequential(ConvBnReLU(self.first_inCh, self.c[0], self.params.s[0]))

        # Layer-1
        self.layer1 = self._make_layer(self.c[0], self.c[1], self.params.t[1], self.params.s[1], self.params.n[1])

        # Layer-2
        self.layer2 = self._make_layer(self.c[1], self.c[2], self.params.t[2], self.params.s[2], self.params.n[2])

        # Layer-3
        self.layer3 = self._make_layer(self.c[2], self.c[3], self.params.t[3], self.params.s[3], self.params.n[3])
        self.layer3_out = nn.Sequential(PointwiseConv(self.c[3], 256))

        # Layer-4
        self.layer4 = self._make_layer(self.c[3], self.c[4], self.params.t[4], self.params.s[4], self.params.n[4])

        # Layer-5
        self.layer5 = self._make_layer(self.c[4], self.c[5], self.params.t[5], self.params.s[5], self.params.n[5])
        self.layer5_out = nn.Sequential(PointwiseConv(self.c[5], 512))

        # Layer-6
        self.layer6 = self._make_layer(self.c[5], self.c[6], self.params.t[6], self.params.s[6], self.params.n[6])

        # Layer-7
        self.layer7 = self._make_layer(self.c[6], self.c[7], self.params.t[7], self.params.s[7], self.params.n[7])

        # Layer-8
        self.layer8 = nn.Sequential(PointwiseConv(self.c[7], self.last_outCh))

        self.out_channels = [256, 512, 1280]

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, inCh, outCh, t, s, n):
        layers = []
        for i in range(n):
            # First layer of each sequence has a stride s and all others use stride 1
            if i == 0:
                layers.append(InvertedResidual(inCh, outCh, t, s))
            else:
                layers.append(InvertedResidual(inCh, outCh, t, 1))

            # Update input channel for next IRB layer in the block
            inCh = outCh
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out52 = self.layer3_out(x)
        x = self.layer4(x)
        x = self.layer5(x)
        out26 = self.layer5_out(x)
        x = self.layer6(x)
        x = self.layer7(x)
        out13 = self.layer8(x)
        return out52, out26, out13

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


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

