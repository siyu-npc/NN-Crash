import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.pooling import MaxPool2d
import torchvision
import torchvision.transforms as transforms
import time
from PIL import Image
import numpy as np
import hiddenlayer
import tensorwatch as tw

img = Image.open('data.jpg')
imgArray = np.array(img)
imgTensor = torch.from_numpy(np.transpose(imgArray, (2, 0, 1))[0]).float() / 255.0
imgTensor = imgTensor.reshape(1, 1, 28, 28)
print(imgTensor.shape)
print(imgTensor.dim())

MnistTrain = torchvision.datasets.FashionMNIST(root='./MNist/', train=True, download=True, transform=transforms.ToTensor())
MnistTest = torchvision.datasets.FashionMNIST(root='./MNist/', train=False, download=True, transform=transforms.ToTensor())
batchSize = 64
TrainIter = torch.utils.data.DataLoader(MnistTrain, batch_size=batchSize, shuffle=True)
TestIter = torch.utils.data.DataLoader(MnistTest, batch_size=batchSize, shuffle=True)

class LeNet(nn.Module) :
    def __init__(self) -> None:
        super(LeNet, self).__init__()
        self.convLayer = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
        )
        self.fcLayer = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),
        )
        
    def forward(self, data) :
        feature = self.convLayer(data)
        output = self.fcLayer(feature.view(data.shape[0], -1))
        return output

device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = LeNet()

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval() # 评估模式, 这会关闭dropout
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train() # 改回训练模式
            n += y.shape[0]
    return acc_sum / n

def train(net, trainIter, testIter, batchSize, optimizer, device, numEpoches) :
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(numEpoches) :
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in trainIter:
            X = X.to(device)
            y = y.to(device)
            yP = net(X)
            lost = loss(yP, y)
            optimizer.zero_grad()
            lost.backward()
            optimizer.step()
            train_l_sum += lost.cpu().item()
            train_acc_sum += (yP.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(testIter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

lr, numEpoches = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train(net, TrainIter, TestIter, batchSize, optimizer, device, numEpoches)


print(net(imgTensor.to(device)))

visGraph = hiddenlayer.build_graph(net, imgTensor.to(device))
visGraph.theme = hiddenlayer.graph.THEMES["blue"].copy()
visGraph.save('./LeNet.py')

tw.draw_model(net, imgTensor.to(device))