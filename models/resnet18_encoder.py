# code modified from segmentation_models_pytorch/encoders/resnet.py


import torch.nn as nn
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from pretrainedmodels.models.torchvision_models import pretrained_settings
import torch.utils.model_zoo as model_zoo


class Res18Encoder(ResNet):
    def __init__(self, pretrained=True):
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2])
        self._depth = 5
        self._out_channels = (3, 64, 64, 128, 256, 512)
        self._in_channels = 3
        del self.fc
        del self.avgpool
        if pretrained:
            self.load_state_dict()

    def get_stage(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        stages = self.get_stage()
        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, **kwargs):
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth')
        state_dict.pop('fc.bias', None)
        state_dict.pop('fc.weight', None)
        super().load_state_dict(state_dict, **kwargs)

    @property
    def out_channels(self):
        return self._out_channels[: self._depth+1]


if __name__ == '__main__':
    import torch
    model = Res18Encoder()
    x = torch.rand(10, 3, 256, 256)
    features = model(x)
    for f in features:
        print(f.shape)

