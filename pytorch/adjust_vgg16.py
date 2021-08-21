import torch
import torch.nn as nn
from torchvision import models

vgg16 = models.vgg16(pretrained=True)
vgg = vgg16.features

for param in vgg.parameters():
    print(param)
    param.requires_grad_(False)