from torch import nn
import torch.nn.functional as F
import torch
import math
import numpy as np


# https://github.com/caojiezhang/MWGAN

class ResidualBlock(nn.Module):
    """
    Residual Block with instance normalization.
    """

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()

        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=False))

    def forward(self, x):
        return (x + self.main(x)) / math.sqrt(2)


class ConvUp(nn.Module):
    def __init__(self, dim_in, dim_out, size=None):
        super().__init__()
        self.conv = nn.utils.spectral_norm(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False))
        self.size = size

    def forward(self, x):
        x = self.conv(x)
        if self.size is None:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            x = F.interpolate(x, size=self.size, mode='bilinear', align_corners=True)
        return x


class DiscriminatorUV(nn.Module):
    """
    Discriminator network with PatchGAN.
    """

    def __init__(self, conv_dim=64, repeat_num=6):
        super(DiscriminatorUV, self).__init__()
        img_size = 256
        max_conv_dim = 512
        dim_in = 2 ** 14 // img_size
        blocks = []
        blocks += [nn.utils.spectral_norm(nn.Conv2d(3, dim_in, 3, 1, 1))]

        repeat_num = int(np.log2(img_size)) - 2
        dim_out = dim_in
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ConvPool(dim_in, dim_out)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.utils.spectral_norm(nn.Conv2d(dim_out, dim_out, (4, 2), 1, 0))]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.utils.spectral_norm(nn.Conv2d(dim_out, 1, 1, 1, 0))]
        self.main = nn.Sequential(*blocks)

    def forward(self, x):
        h = self.main(x)
        return h.squeeze()


class Discriminator(nn.Module):
    """
    Discriminator network with PatchGAN.
    """

    def __init__(self, conv_dim=64, repeat_num=6):
        super(Discriminator, self).__init__()
        img_size = 256
        max_conv_dim = 512
        dim_in = 2 ** 14 // img_size
        blocks = []
        blocks += [nn.utils.spectral_norm(nn.Conv2d(3, dim_in, 3, 1, 1))]

        repeat_num = int(np.log2(img_size)) - 2
        dim_out = dim_in
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ConvPool(dim_in, dim_out)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.utils.spectral_norm(nn.Conv2d(dim_out, dim_out, 4, 1, 0))]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.utils.spectral_norm(nn.Conv2d(dim_out, 1, 1, 1, 0))]
        self.main = nn.Sequential(*blocks)

    def forward(self, x):
        h = self.main(x)
        return h.squeeze()


class Conv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.utils.spectral_norm(nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding, bias=bias))
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class ConvPool(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.utils.spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        self.relu = nn.LeakyReLU(0.2)
        self.down = nn.AvgPool2d(2)
        self.norm = nn.InstanceNorm2d(dim_in, affine=True, track_running_stats=False)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.down(x)
        return x


class ResEncoder(nn.Module):
    """
    Encoder network.
    """

    def __init__(self, in_ch=3, conv_dim=64, repeat_num=3):
        super(ResEncoder, self).__init__()
        layers = []
        layers.append(
            nn.utils.spectral_norm(nn.Conv2d(in_ch, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=False))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.utils.spectral_norm(
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False)))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=False))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        out = self.main(x)
        return out


class ResDecoder(nn.Module):
    def __init__(self, conv_dim=64, repeat_num=3):
        super(ResDecoder, self).__init__()

        layers = []
        # downsampling 2^2
        curr_dim = conv_dim * 4
        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            # layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(ConvUp(curr_dim, curr_dim // 2))
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=False))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.utils.spectral_norm(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False)))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, h):
        out = self.main(h)
        return out


class ConvDown(nn.Module):
    def __init__(self, in_ch, out_ch, k_sz):
        super().__init__()
        self.main = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, k_sz, 2, bias=False)),
                                  nn.InstanceNorm2d(out_ch, affine=True, track_running_stats=False),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        return self.main(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = ResEncoder()
        self.decoder = ResDecoder()

    def forward(self, uv_map):
        code = self.encoder(uv_map)
        out = self.decoder(code)
        return out


if __name__ == '__main__':
    # model = DiscriminatorUVTotal()
    model = DiscriminatorUV()
    I = torch.rand(10, 3, 256, 128)
    out = model(I)
    print(out.shape)
