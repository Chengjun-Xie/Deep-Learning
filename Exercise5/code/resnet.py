import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        """
        For the 18-layer variant of ResNet,
        each ResBlock consists of a sequence of (Conv2D, BatchNorm, ReLU) that is repeated twice.
        :param in_channel:
        :param out_channel: the number of output channels for Conv2D is given by the argument out channels.
        :param stride:  The stride of the ﬁrst Conv2D is given by stride. For the second convolution, no stride is used.
        """
        super(ResBlock, self).__init__()
        # we recommend to apply a batchnorm layer before you add the result to the output.
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channel)
        )

        #  the size and number of channels needs to be adapted.
        #  To this end, we recommend to apply a 1×1 convolution
        #  to the input with stride and channels set accordingly
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1,
                          stride=stride),
                nn.BatchNorm2d(num_features=out_channel))

    def forward(self, x):
        y = self.block(x)
        y += self.shortcut(x)
        y = F.relu(y)
        return y


class ResNet(nn.Module):
    def __init__(self, ResBlock):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())

        self.conv2_x = nn.Sequential(nn.MaxPool2d(3, 2),
                                     ResBlock(64, 64, 1),
                                     ResBlock(64, 64, 1))

        self.conv3_x = nn.Sequential(ResBlock(64, 128, 2),
                                     ResBlock(128, 128, 1))

        self.conv4_x = nn.Sequential(ResBlock(128, 256, 2),
                                     ResBlock(256, 256, 1))

        self.conv5_x = nn.Sequential(ResBlock(256, 512, 2),
                                     ResBlock(512, 512, 1))

        self.pool_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2_x(y)
        y = self.conv3_x(y)
        y = self.conv4_x(y)
        y = self.conv5_x(y)

        y = self.pool_avg(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y


def ResNet18():
    return ResNet(ResBlock)






