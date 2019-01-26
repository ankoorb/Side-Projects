import torch
import torch.nn as nn
import torch.nn.functional as F

from base.network import MobileNet


# Atrous Spatial Pyramid Pooling
# ------------------------------
class AtrousConvBnRelu(nn.Module):
    """
    [Atrous CONV]-[BN]-[ReLU]
    """
    def __init__(self, inCh, outCh, dilation=1):
        super(AtrousConvBnRelu, self).__init__()
        self.inCh = inCh
        self.outCh = outCh
        self.dilation = dilation
        self.kernel = 1 if self.dilation == 1 else 3
        self.padding = 0 if self.dilation == 1 else self.dilation
        self.atrous_conv = nn.Sequential(
            nn.Conv2d(self.inCh, self.outCh, self.kernel, stride=1,
                      padding=self.padding, dilation=self.dilation, bias=False),
            nn.BatchNorm2d(self.outCh),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.atrous_conv(x)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling

    Ref(s): https://github.com/rishizek/tensorflow-deeplab-v3-plus/blob/master/deeplab_model.py
    and https://github.com/chenxi116/DeepLabv3.pytorch/blob/master/deeplab.py
    """
    def __init__(self, inCh, outCh):
        super(ASPP, self).__init__()
        self.rates = [1, 6, 12, 18] # for output stride 16
        self.inCh = inCh
        self.outCh = outCh

        # ASPP layers
        # (a) One 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18)
        self.conv_1x1_0 = AtrousConvBnRelu(inCh=self.inCh, outCh=self.outCh,
                                           dilation=self.rates[0])
        self.conv_3x3_1 = AtrousConvBnRelu(inCh=self.inCh, outCh=self.outCh,
                                           dilation=self.rates[1])
        self.conv_3x3_2 = AtrousConvBnRelu(inCh=self.inCh, outCh=self.outCh,
                                           dilation=self.rates[2])
        self.conv_3x3_3 = AtrousConvBnRelu(inCh=self.inCh, outCh=self.outCh,
                                           dilation=self.rates[3])

        # (b) The image-level features
        # Global Average Pooling
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # CONV-BN-ReLU after Global Average Pooling
        self.conv_bn_relu_4 = nn.Sequential(
            nn.Conv2d(self.inCh, self.outCh, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.outCh),
            nn.ReLU(inplace=True)
        )

        # CONV-BN-ReLU after Concatenation. NOTE: 5 Layers are concatenated
        self.conv_bn_relu_5 = nn.Sequential(
            nn.Conv2d(self.outCh * 5, self.outCh, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.outCh),
            nn.ReLU(inplace=True)
        )

        self._initialize_weights()

    def forward(self, x):
        x0 = self.conv_1x1_0(x)  # size: [1, outCh, fs, fs]
        x1 = self.conv_3x3_1(x)  # size: [1, outCh, fs, fs]
        x2 = self.conv_3x3_2(x)  # size: [1, outCh, fs, fs]
        x3 = self.conv_3x3_3(x)  # size: [1, outCh, fs, fs]

        # Global Average Pooling, CONV-BN-ReLU and upsample
        global_avg_pool = self.global_avg_pooling(x)

        x4 = self.conv_bn_relu_4(global_avg_pool)

        upsample = F.interpolate(x4, size=(x.size(2), x.size(3)), mode='bilinear',
                                 align_corners=True)

        # Concatinate
        x_concat = torch.cat([x0, x1, x2, x3, upsample], dim=1) # size: [1, 5 * outCh, fs, fs]

        # CONV-BN-ReLU after concatination
        out = self.conv_bn_relu_5(x_concat)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# Decoder
# -------
class Decoder(nn.Module):
    """
    Decoder for DeepLabV3+
    """

    def __init__(self, low_level_inch, low_level_outch, inCh, outCh, n_classes):
        super(Decoder, self).__init__()
        self.low_level_inch = low_level_inch
        self.low_level_outch = low_level_outch  # 48 (or lower for speed)
        self.inCh = inCh
        self.outCh = outCh
        self.n_classes = n_classes

        # 1x1 Conv with BN and ReLU for low level features
        self.conv_1x1_bn_relu = nn.Sequential(
            nn.Conv2d(self.low_level_inch, self.low_level_outch, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.low_level_outch),
            nn.ReLU(inplace=True)
        )

        # Conv block with BN and ReLU (paper suggests to use a few 3x3 Convs, but using only 1
        # for speed improvement) and final Conv 1x1
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.inCh + self.low_level_outch, self.outCh, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(self.outCh),
            nn.ReLU(inplace=True),

            # For reducing number of channels
            nn.Conv2d(self.outCh, self.n_classes, kernel_size=1, stride=1, bias=False)
        )

        self._initialize_weights()

    def forward(self, x, low_level_features):

        # Low level features from MobileNetV2
        low_level_features = self.conv_1x1_bn_relu(low_level_features)

        # Upsample features from ASPP by 4
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)

        # Concatinate
        x_concat = torch.cat([x, low_level_features], dim=1)

        # Final Convolution
        out = self.conv_block(x_concat)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# DeepLabV3+
# ----------
class DeepLabV3Plus(nn.Module):
    def __init__(self, config):
        super(DeepLabV3Plus, self).__init__()
        self.config = config

        # Base Network
        self.base = MobileNet(weight_file=self.config.pretrained_weights, params=self.config)

        # ASPP Module
        self.aspp = ASPP(inCh=self.config.aspp_inch,
                         outCh=self.config.aspp_outch)

        # Decoder
        self.decoder = Decoder(low_level_inch=self.config.low_level_inCh,
                               low_level_outch=self.config.low_level_outCh,
                               inCh=self.config.in_channels,
                               outCh=self.config.out_channels,
                               n_classes=self.config.n_classes)

    def forward(self, x):
        # Extract features from base network
        base_out, low_level_features = self.base(x)

        # Pool base network output using Atrous Spatial Pyramid Pooling
        aspp_out = self.aspp(base_out)

        # Use decoder to obtain object boundaries
        decoder_out = self.decoder(aspp_out, low_level_features)

        # Upsample features from decoder by 4
        out = F.interpolate(decoder_out, scale_factor=4, mode='bilinear', align_corners=True)

        return out

