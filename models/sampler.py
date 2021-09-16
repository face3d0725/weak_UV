import torch
import torch.nn.functional as F
import torch.nn as nn
from models.resnet18_encoder import Res18Encoder
from models.fpn_decoder import FPNDecoder, SegmentationHead


class SamplerHead(nn.Module):
    def __init__(self, sz=256):
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),

        )
        self.sz = sz
        self.loc = nn.Sequential(nn.Linear(512, 256),
                                 nn.ReLU(True),
                                 nn.Linear(256, 128),
                                 nn.ReLU(True),
                                 nn.Linear(128, self.sz ** 2 * 2))
        self.initialize()

    def forward(self, feat):
        feat = self.pool(feat)
        feat = feat.view(-1, 512)
        xym = self.loc(feat)  # x, y, mask
        xx, yy = xym[:, :self.sz ** 2], xym[:, self.sz ** 2:self.sz ** 2 * 2]
        xx, yy = xx.view(-1, self.sz, self.sz), yy.view(-1, self.sz, self.sz)
        grid = torch.stack((xx, yy), 3)
        return grid

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.loc[4].weight.data.zero_()
        theta = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32).unsqueeze(0)  # initialize with identity grid
        grid = F.affine_grid(theta, size=[1, 3, self.sz, self.sz], align_corners=True)
        X = grid[0, ..., 0].view(-1)
        Y = grid[0, ..., 1].view(-1)
        XY = torch.cat((X, Y), 0)
        self.loc[4].bias.data.copy_(XY)


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Res18Encoder()
        self.decoder = FPNDecoder(self.encoder.out_channels)
        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=1,  # only one class
            kernel_size=1,
            upsampling=4
        )
        self.sampler_head = SamplerHead()

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        grid_sample = self.sampler_head(features[-1])
        return grid_sample, masks


if __name__ == '__main__':
    model = Sampler()
    x = torch.rand(10, 3, 256, 256)
    masks, grid_sample = model(x)
    print(masks.shape)
    print(grid_sample.shape)
