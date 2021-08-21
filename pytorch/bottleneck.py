import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm1d

class Bottleneck(nn.Module) :
    def __init__(self, in_dim, out_dim, stride=1):
        super().__init__()
        self.bottlenect = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, 3, stride, 1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, 1),
            nn.BatchNorm2d(out_dim)
        )

    def forward(self, x) :
        identity = x
        out = self.bottlenect(x)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

if __name__ == '__main__' :
    bottlenect1 = Bottleneck(64, 256).cuda()
    input = torch.randn(1, 64, 56, 56).cuda()
    output = bottlenect1(input)
    # print(output.shape)
    downsample = nn.Conv2d(1, 3, 1, 1)
    test = torch.randn(1, 1, 4, 4)
    print(test)
    print("=======================")
    print(downsample(test))
