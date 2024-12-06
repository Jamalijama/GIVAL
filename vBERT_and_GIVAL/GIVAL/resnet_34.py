import torch
from torch import nn
from torch.nn import functional as F
import pickle

class ResidualBlock(nn.Module):
    # Implement submodule: Residual Block

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (3, 3), (stride, stride), 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, (3, 3), (1, 1), 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet(nn.Module):
    """
    Implement submodule：ResNet18 or ResNet34
    ResNet18 contains multiple layers, each of which contains multiple residual blocks
    Use submodules to implement residual blocks and _make_layer functions to implement layers
    """

    def __init__(self, blocks, num_classes=1000, begin=3, dropout=0.2):
        super(ResNet, self).__init__()
        self.model_name = 'resnet'

        # First few layers: size conversion, start with 3 channels
        self.pre = nn.Sequential(
            nn.Conv2d(begin, 64, (7, 7), (2, 2), 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))

        # Duplicate layers with residual blocks
        self.layer1 = self._make_layer(64, 64, blocks[0])
        self.layer2 = self._make_layer(64, 128, blocks[1], stride=2)
        self.layer3 = self._make_layer(128, 256, blocks[2], stride=2)
        self.layer4 = self._make_layer(256, 512, blocks[3], stride=2)
        self.dropout = nn.Dropout(p=dropout)
        # Full connection for classification
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        # Build layers, containing multiple residual blocks
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1), (stride, stride), bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
        )

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        len_x = int(x.shape[0])
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x_FC = []

        for i in range(len_x):
            x_sample = x[i]
            x_sample_FC = []
            for x0 in x_sample:
                x0= float(x0)
                x_sample_FC.append(x0)
            x_FC.append(x_sample_FC)
        file1 = open('FC_data_test.pkl','wb')
        pickle.dump(x_FC,file1)
        print(len(x_FC))
        print(len(x_FC[0]))
        return self.fc(x)


def ResNet34(num_classes, begin, dropout=0.2):
    return ResNet([3, 4, 6, 3], num_classes, begin, dropout)


if __name__ == '__main__':
    # model = torchvision.models.resnet50()
    model = ResNet34(2, 3)
    print(model)

    input = torch.randn(1, 3, 128, 128)
    out = model(input)
    print(out.shape)
    print(out)
