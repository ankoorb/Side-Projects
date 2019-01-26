"""
Creates a MobileNetV2 model as defined in the paper: M. Sandler, 
A. Howard, M. Zhu, A. Zhmoginov, L.-C. Chen. "MobileNetV2: Inverted 
Residuals and Linear Bottlenecks.", arXiv:1801.04381, 2018."

Code reference: https://github.com/tonylins/pytorch-mobilenet-v2
ImageNet pretrained weights: https://drive.google.com/file/d/1jlto6HRVD3ipNkAl1lNhDbkBp7HylaqR
"""
import math
import torch
import torch.nn as nn


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

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
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
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


# Encoder for Show Attend and Tell
class EncoderCNN(nn.Module):
    """
    Convolutional Neural Network (MobileNetV2) that encodes input image 
    into encoded feature representations.
    """

    def __init__(self, weight_file, feature_size=14, tune_layer=None, finetune=False):
        """
        Parameters
        ----------
        weight_file: str, path to MobileNetV2 pretrained weights.
        feature_size: int, encoded feature map size to be used.
        tune_layer: int, tune layers from this layer onwards. For
            MobileNetV2 select integer from 0 (early) to 18 (final)
        finetune: bool, fine tune layers
        """
        super(EncoderCNN, self).__init__()
        self.weight_file = weight_file
        self.feature_size = feature_size
        self.tune_layer = tune_layer
        self.finetune = finetune
        self.pretrained = True

        # MobileNetV2 pretrained on ImageNet
        cnn = MobileNet(pretrained=self.pretrained, weight_file=self.weight_file)

        # Remove classification layer
        modules = list(cnn.children())[:-1]
        self.cnn = nn.Sequential(*modules)

        # Resize feature maps to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.feature_size, self.feature_size))

        # Fine-tune
        self.fine_tune()

    def forward(self, images):
        """
        Parameters
        ----------
        images: PyTorch tensor, size: [M, 3, H, W]
        """
        features = self.cnn(images)  # size: [M, 1280, H/32, W/32]
        features = self.adaptive_pool(features)  # size: [M, 1280, fs, fs]
        features = features.permute(0, 2, 3, 1)  # size: [M, fs, fs, 1280]
        return features

    def fine_tune(self):
        """
        Fine-tuning CNN.
        """
        # Disable gradient computation
        for param in self.cnn.parameters():
            param.requires_grad = False

        # Enable gradient computation for few layers
        for child in list(self.cnn.children())[0][self.tune_layer:]:
            for param in child.parameters():
                param.requires_grad = self.finetune

