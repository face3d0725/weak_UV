'''
    implement Light CNN
    @author: Alfred Xiang Wu
    @date: 2017.07.04
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=kernel_size, stride=stride,
                                    padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2 * out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x


class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out


class network_29layers_v2(nn.Module):
    def __init__(self, block, layers, num_classes=79077):
        super(network_29layers_v2, self).__init__()
        self.conv1 = mfm(1, 48, 5, 1, 2)
        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.group1 = group(48, 96, 3, 1, 1)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.group2 = group(96, 192, 3, 1, 1)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.group3 = group(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.group4 = group(128, 128, 3, 1, 1)
        self.fc = nn.Linear(8 * 8 * 128, 256)

    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    @torch.no_grad()
    def rgb2gray(self, x):
        weight = torch.Tensor([0.3, 0.59, 0.11]).view(1, 3, 1, 1).to(x.device)
        weight = weight.expand(x.shape[0], 3, 1, 1)
        x_gray = (weight*x).sum(1, keepdim=True)
        return x_gray


    def forward(self, x):
        x = self.rgb2gray(x)
        x = self.conv1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block1(x)
        x = self.group1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block2(x)
        x = self.group2(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = x.view(x.size(0), -1)
        fc = self.fc(x)
        fc_norm = torch.norm(fc, dim=1, keepdim=True)
        fc = fc/fc_norm
        return fc


def LightCNN_29Layers_v2(**kwargs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = network_29layers_v2(resblock, [1, 2, 3, 4], **kwargs)
    model = torch.nn.DataParallel(model.to(device), [0, 1])
    state_dict = torch.load('../weights/LightCNN_29Layers_V2_checkpoint.pth.tar', map_location=torch.device('cpu'))[
        'state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'fc2' not in k:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    return model.eval()


if __name__ == '__main__':
    model = LightCNN_29Layers_v2()
    x = torch.rand(10, 3, 128, 128).cuda()
    out = model(x)
    print(out.shape)
    print(out[0])
