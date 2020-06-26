import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm

class UNet(nn.Module):
    training_data = []
    INPUT_SIZE = 572

    def __init__(self):
        super().__init__()
        self.down_conv_1 = self.double_conv3x3(1, 64)
        self.max_pool_1 = nn.MaxPool2d(2, 2)
        self.down_conv_2 = self.double_conv3x3(64, 128)
        self.max_pool_2 = nn.MaxPool2d(2, 2)
        self.down_conv_3 = self.double_conv3x3(128, 256)
        self.max_pool_2 = nn.MaxPool2d(2, 2)
        self.down_conv_4 = self.double_conv3x3(256, 512)
        self.max_pool_2 = nn.MaxPool2d(2, 2)
        self.down_conv_5 = self.double_conv3x3(512, 1024)
        self.up_pool_1 = nn.ConvTranspose2d(1024, 1024, 2)
        self.up_conv_1 = self.double_conv3x3(1024, 512)
        self.up_pool_2 = nn.ConvTranspose2d(512, 512, 2)
        self.up_conv_2 = self.double_conv3x3(512, 256)
        self.up_pool_3 = nn.ConvTranspose2d(256, 256, 2)
        self.up_conv_3 = self.double_conv3x3(256, 128)
        self.up_pool_4 = nn.ConvTranspose2d(128, 128, 2)
        self.up_conv_4 = self.double_conv3x3(128, 64)
        self.final_conv = nn.Conv2d(64, 1, 1)

    def forward(self, img):
        x = self.down_conv_1(img)
        x = self.max_pool_1(x)
        x = self.down_conv_2(x)
        x = self.max_pool_2(x)
        x = self.down_conv_3(x)
        x = self.max_pool_2(x)
        x = self.down_conv_4(x)
        x = self.max_pool_2(x)
        x = self.down_conv_5(x)
        x = self.up_pool_1(x)
        x = self.up_conv_1(x)
        x = self.up_pool_2(x)
        x = self.up_conv_2(x)
        x = self.up_pool_3(x)
        x = self.up_conv_3(x)
        x = self.up_pool_4(x)
        x = self.up_conv_4(x)
        x = self.final_conv(x)

        return x

    def double_conv3x3(self, input, output):
        conv = nn.Sequential(
            nn.Conv2d(input, output, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(output, output,3),
            nn.ReLU(inplace=True)
        )
        return conv

net = UNet()
pic = torch.randn(net.INPUT_SIZE, net.INPUT_SIZE).view(-1, 1, net.INPUT_SIZE, net.INPUT_SIZE)
print(net(pic))