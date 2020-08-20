import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import torch.optim as optim

INPUT_SIZE_ORG = 572
OUTPUT_SIZE_ORG = 388

class UNet(nn.Module):
    training_data = []
    INPUT_SIZE = INPUT_SIZE_ORG
    OUTPUT_SIZE = OUTPUT_SIZE_ORG
    MODEL_NAME = "model/UNet_mdl"

    def __init__(self):
        super().__init__()

        # ENCRYPTOR
        self.down_conv_1 = self.double_conv3x3(1, 32)
        self.max_pool_1 = nn.MaxPool2d(2, 2, return_indices=False)
        self.down_conv_2 = self.double_conv3x3(32, 64)
        self.max_pool_2 = nn.MaxPool2d(2, 2, return_indices=False)
        self.down_conv_3 = self.double_conv3x3(64, 128)
        self.max_pool_3 = nn.MaxPool2d(2, 2, return_indices=False)
        self.down_conv_4 = self.double_conv3x3(128, 256)
        self.max_pool_4 = nn.MaxPool2d(2, 2, return_indices=False)
        self.down_conv_5 = self.double_conv3x3(256, 512)

        # DECRYPTER
        self.up_pool_1 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2
        )
        self.up_conv_1 = self.double_conv3x3(512, 256)
        self.up_pool_2 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2
        )
        self.up_conv_2 = self.double_conv3x3(256, 128)
        self.up_pool_3 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2
        )
        self.up_conv_3 = self.double_conv3x3(128, 64)
        self.up_pool_4 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=2,
            stride=2
        )
        self.up_conv_4 = self.double_conv3x3(64, 32)
        self.final_conv = nn.Conv2d(32, 1, 1)
        self.sigm = nn.Sigmoid()

        self.loss_function = nn.MSELoss(reduction='sum')
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def crop_tensor(self, tensor, target_tensor):
        target_size = target_tensor.size()[2]
        tensor_size = tensor.size()[2]
        d = tensor_size - target_size
        d = d // 2
        return tensor[:, :, d:tensor_size-d, d:tensor_size-d]

    def forward(self, img):
        # ENCRYPTOR
        x1 = self.down_conv_1(img)
        x2 = self.max_pool_1(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_4(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.max_pool_4(x7)
        x9 = self.down_conv_5(x8)

        # DECRYPTER
        x = self.up_pool_1(x9)
        y = self.crop_tensor(x7, x)
        x = self.up_conv_1(torch.cat([x, y], 1))

        x = self.up_pool_2(x)
        y = self.crop_tensor(x5, x)
        x = self.up_conv_2(torch.cat([x, y], 1))

        x = self.up_pool_3(x)
        y = self.crop_tensor(x3, x)
        x = self.up_conv_3(torch.cat([x, y], 1))

        x = self.up_pool_4(x)
        y = self.crop_tensor(x1, x)
        x = self.up_conv_4(torch.cat([x, y], 1))

        x = self.final_conv(x)
        x = self.sigm(x)

        return x

    def double_conv3x3(self, input, output):
        conv = nn.Sequential(
            nn.Conv2d(input, output, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(output, output,3),
            nn.ReLU(inplace=True)
        )
        return conv

# net = UNet()
# pic = torch.randn(net.INPUT_SIZE, net.INPUT_SIZE).view(-1, 1, net.INPUT_SIZE, net.INPUT_SIZE)
# print(net(pic))