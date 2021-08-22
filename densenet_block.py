import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module) :
    def __init__(self, nChannels, growthRate):
        super().__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3, padding=1, bias=False)
    def forward(self, x) :
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class DenseBlock(nn.Module) :
    def __init__(self, nChannels, growthRate, nDenseBlocks):
        super().__init__()
        layers = []
        for i in range(int(nDenseBlocks)) :
            layers.append(Bottleneck(nChannels, growthRate))
            nChannels += growthRate
        self.denseblock = nn.Sequential(*layers)
    def forward(self, x):
        return self.denseblock(x)

if __name__ == '__main__' :
    denseblock = DenseBlock(64, 32, 6).cuda()
    input = torch.randn(1, 64, 256, 256).cuda()
    output = denseblock(input)
    print(output.shape)