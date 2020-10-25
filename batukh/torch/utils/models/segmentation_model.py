from torchvision.models.resnet import ResNet, Bottleneck
from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
import os
from os.path import join

# Data Prep

# Remove this??
import numpy as np


# Transforms


# Models


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ModifiedBottleneck(Bottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, is_last=False):

        super(ModifiedBottleneck, self).__init__(inplanes, planes, stride, downsample, groups,
                                                 base_width, dilation, norm_layer)

        self.is_last = is_last

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if not self.is_last:
            out += identity
        out = self.relu(out)

        return out


class ResNetLayer(nn.Module):
    def __init__(self, block, inplanes, planes, blocks, sec_output_block=None, strides=None, connect_last=False):
        super(ResNetLayer, self).__init__()

        if strides is None:
            strides = [1]*blocks

        self.blocks = blocks
        self.sec_output_block = sec_output_block
        planes = int(planes/block.expansion)

        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(planes * block.expansion, eps=1e-05,
                           momentum=0.1, affine=True, track_running_stats=True)
        )

        self.add_module("0", block(inplanes, planes, strides[0], downsample))
        inplanes = int(planes * block.expansion)

        for i in range(1, blocks-1):
            self.add_module(str(i), block(inplanes, planes, strides[i]))
        i += 1
        if connect_last:
            is_last = False
        else:
            is_last = True
        self.add_module(str(i), block(
            inplanes, planes, strides[i], is_last=is_last))

    def forward(self, x):

        for i in range(self.blocks):
            block = getattr(self, str(i))
            x = block(x)
            if i == self.sec_output_block:
                sec_output = x.clone()
        if self.sec_output_block is not None:
            return x, sec_output
        return x


class MyResNet(nn.Module):
    def __init__(self, block, layers, strides=None, norm_layer=None, sec_output_blocks=None):
        super(MyResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if strides is None:
            strides[None]*len(layers)
        if sec_output_blocks is None:
            sec_output_blocks = [None]*layers

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = ResNetLayer(block, self.inplanes, self.inplanes*block.expansion, layers[0],
                                  strides=strides[0], sec_output_block=sec_output_blocks[0])

        self.inplanes = self.inplanes*block.expansion

        self.layer2 = ResNetLayer(block, self.inplanes, self.inplanes*block.expansion//2, layers[1],
                                  strides=strides[1], sec_output_block=sec_output_blocks[1])

        self.inplanes = self.inplanes*block.expansion//2

        self.layer3 = ResNetLayer(block, self.inplanes, self.inplanes*block.expansion//2, layers[2],
                                  strides=strides[2], sec_output_block=sec_output_blocks[2])

        self.inplanes = self.inplanes*block.expansion//2

        self.layer4 = ResNetLayer(block, self.inplanes, self.inplanes*block.expansion//2, layers[3],
                                  strides=strides[3], sec_output_block=sec_output_blocks[3])

    def forward(self, x):
        sec_outputs = []

        sec_outputs.append(x.clone())

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        sec_outputs.append(x.clone())

        x = self.maxpool(x)

        x, sec = self.layer1(x)
        sec_outputs.append(sec)

        x, sec = self.layer2(x)
        sec_outputs.append(sec)

        x, sec = self.layer3(x)
        sec_outputs.append(sec)

        x = self.layer4(x)

        return x, sec_outputs


class UpScalerUnit(nn.Module):
    def __init__(self, n_current, n_output, n_copy):
        super(UpScalerUnit, self).__init__()
        # self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = nn.Conv2d(n_current+n_copy, n_output,
                              kernel_size=(3, 3), padding=1, bias=False)

    def forward(self, x, copy):
        # print(x.shape, copy.shape)

        # Dynamically created a layer to match the size
        self.upsample = nn.UpsamplingNearest2d(
            size=(copy.shape[-2], copy.shape[-1]))
        x = self.upsample(x)
        x = torch.cat((copy, x), dim=1)
        x = self.conv(x)
        return x


class UpScaler(nn.Module):
    def __init__(self, n_classes):
        super(UpScaler, self).__init__()

        self.upscale1 = UpScalerUnit(512, 512, 512)

        self.upscale2 = UpScalerUnit(512, 256, 512)

        self.upscale3 = UpScalerUnit(256, 128, 256)

        self.upscale4 = UpScalerUnit(128, 64, 64)

        self.upscale5 = UpScalerUnit(64, 32, 3)

        self.conv2 = nn.Conv2d(32, n_classes, kernel_size=1, bias=False)

    def forward(self, x, copies):

        x = self.upscale1(x, copies.pop(-1))
        x = self.upscale2(x, copies.pop(-1))
        x = self.upscale3(x, copies.pop(-1))
        x = self.upscale4(x, copies.pop(-1))
        x = self.upscale5(x, copies.pop(-1))

        x = self.conv2(x)

        return x


class SegmentationModel(nn.Module):
    def __init__(self, use_pretrained=True, lock_pretrained=True):
        super(SegmentationModel, self).__init__()

        self.downsampler = MyResNet(ModifiedBottleneck, [3, 4, 6, 3],
                                    strides=[[1, 1, 2], [1, 1, 1, 2],
                                             [1, 1, 1, 1, 1, 2], None],
                                    sec_output_blocks=[1, 2, 4, None])

        if use_pretrained:
            resnet = models.resnet50(pretrained=True)
            resnet.avgpool = Identity()
            resnet.fc = Identity()
            print(self.downsampler.load_state_dict(resnet.state_dict()))

            if lock_pretrained:
                for param in self.downsampler.parameters():
                    param.requires_grad = False

        self.conv1 = nn.Conv2d(1024, 512, kernel_size=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(2048, 512, kernel_size=(1, 1), bias=False)
        self.upscaler = UpScaler(2)

    def forward(self, x):
        x, sec_outputs = self.downsampler(x)
        sec_outputs[-1] = self.conv1(sec_outputs[-1])
        x = self.conv2(x)
        x = self.upscaler(x, sec_outputs)

        return x
