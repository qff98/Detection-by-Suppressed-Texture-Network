"""
Ported to pytorch thanks to [tstandley](https://github.com/tstandley/Xception-PyTorch)

@author: tstandley
Adapted by cadene

Creates an Xception Model as defined in:

Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf

This weights ported from the Keras implementation. Achieves the following performance on the validation set:

Loss:0.9173 Prec@1:78.892 Prec@5:94.292

REMEMBER to set your image size to 3x299x299 for both test and validation

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])

The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from attention import CBAM

class AMTEN(nn.Module):

    def __init__(self):
        super(AMTEN, self).__init__()
        self.AMTENconv1 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.AMTENconv2_1 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.AMTENconv2_2 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.AMTENconv3_1 = nn.Conv2d(6, 6, 3, 1, 1, bias=True)
        self.AMTENconv3_2 = nn.Conv2d(6, 6, 3, 1, 1, bias=True)

    def forward(self, x):
        x_feature33 = self.AMTENconv1(x)
        xfea3 = x_feature33-x
        x1 = xfea3
        x2 = self.AMTENconv2_1(x1)
        x2 = self.AMTENconv2_2(x2)
        xcat_1 = torch.cat((x2, x1), dim=1)
        x3 = self.AMTENconv3_1(xcat_1)
        x3 = self.AMTENconv3_2(x3)
        xcat_2 = torch.cat((x3, x1,xcat_1), dim=1)
        # print(xcat_2.shape)
        return xcat_2


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class REPBLOCK(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', use_se=False,active = True):
        super(REPBLOCK, self).__init__()
        self.groups = groups
        self.in_channels = in_channels
        assert kernel_size == 3
        assert padding == 1


        padding_11 = padding - kernel_size // 2
        if active == True:
            self.nonlinearity = nn.ReLU()
        else :
            self.nonlinearity = nn.Identity()

        if use_se:
            self.se = nn.Identity()
        else:
            self.se = nn.Identity()

        # self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
        # self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None
        self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
        self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)

    def forward(self, inputs):
        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs)))


class REAMTEN(nn.Module):

    def __init__(self,fea = False):
        super(REAMTEN, self).__init__()
        self.fea = fea
        self.feature33 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.feature11 = nn.Conv2d(3, 3, 1, 1, 0, bias=True)
        # self.bn1 = nn.BatchNorm2d(3)
        # self.bn2 = nn.BatchNorm2d(3)

        # self.conv1 = nn.Conv2d(3,3,3,1,1,bias=False)
        # self.conv11 = nn.Conv2d(3, 3, 3, 1, 1, bias=False)
        # self.bn1 = nn.BatchNorm2d(3)
        # self.bn11 = nn.BatchNorm2d(3)
        # self.conv2 = nn.Conv2d(6, 6, 3, 1, 1, bias=False)
        # self.conv22 = nn.Conv2d(6, 6, 3, 1, 1, bias=False)
        # self.bn2 = nn.BatchNorm2d(6)
        # self.bn22 = nn.BatchNorm2d(6)
        # self.conv3 = nn.Conv2d(12, 12, 3, 1, 1, bias=False)
        # self.conv33 = nn.Conv2d(12, 12, 3, 1, 1, bias=False)
        # self.bn3 = nn.BatchNorm2d(12)
        # self.bn33 = nn.BatchNorm2d(12)
        # self.conv4 = nn.Conv2d(24, 24, 3, 1, 1, bias=False)
        # self.conv44 = nn.Conv2d(24, 24, 3, 1, 1, bias=False)
        # self.bn4 = nn.BatchNorm2d(24)
        # self.relu = nn.ReLU(inplace=True)

        self.ca1 = CBAM(3,ratio=3)
        self.ca3 = CBAM(6,ratio=6)
        self.ca4 = CBAM(12,ratio=12)
        # self.ca5 = CBAM(24)

        self.conv1 = REPBLOCK(3, 3, 3, 1, 1, active=True)
        self.conv11 = REPBLOCK(3,3,3,1,1,active=True)
        self.conv2 = REPBLOCK(6,6,3,1,1,active=True)
        self.conv22 = REPBLOCK(6,6,3,1,1,active=True)
        self.conv3 = REPBLOCK(12, 12, 3, 1, 1, active=True)
        self.conv33 = REPBLOCK(12, 12, 3, 1, 1, active=True)
        # self.conv4 = REPBLOCK(24, 24, 3, 1, 1, active=True)
    def forward(self,x):
        x_feature33 = self.feature33(x)
        xfea3 = x_feature33 - x
        # xfea3 = self.bn1(xfea3)
        #
        x_feature11 = self.feature11(x)
        xfea1 = x_feature11 - x
        # xfea1 = self.bn2(xfea1)

        x1 = xfea3+xfea1

        x2 = self.conv1(x1)
        x2 = self.conv11(x2)
        x2 = self.ca1(x2)
        xcat_1 = torch.cat((x2, x1), dim=1)
        x3 = self.conv2(xcat_1)
        x3 = self.conv22(x3)
        x3 = self.ca3(x3)
        xcat_2 = torch.cat((x3, xcat_1), dim=1)

        x4 = self.conv3(xcat_2)
        x4 = self.conv33(x4)
        x4 = self.ca4(x4)

        # x5 = self.conv4(xcat_3)

        # x2 = self.relu(self.bn1(self.conv1(x1)))
        # x2 = self.relu(self.bn11(self.conv11(x2)))
        # x2 = self.ca1(x2)
        # xcat_1 = torch.cat((x2, x1), dim=1)
        # x3 = self.relu(self.bn2(self.conv2(xcat_1)))
        # x3 = self.relu(self.bn22(self.conv22(x3)))
        # x3 = self.ca3(x3)
        # xcat_2 = torch.cat((x3, xcat_1), dim=1)
        # x4 = self.relu(self.bn3(self.conv3(xcat_2)))
        # x4 = self.relu(self.bn33(self.conv33(x4)))
        # x4 = self.ca4(x4)
        # xcat_3 = torch.cat((x4, xcat_2), dim=1)
        # x5 = self.relu(self.bn4(self.conv4(xcat_3)))
        # x5 = self.ca5(x5)
        # print(xcat_2.shape)
        out = x4
        return out




