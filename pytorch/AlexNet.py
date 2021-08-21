import torch
from torch.functional import Tensor
import torch.nn as nn
from torch.nn.modules.activation import Softmax
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
from PIL import Image
import numpy as np
import os
import hiddenlayer
from torchvision.transforms.transforms import ToTensor

print(torch.__version__)

def loadData(batchSize, resize) :
    trans = []
    if resize :
        trans.append(transforms.Resize((resize, resize)))
    trans.append(transforms.ToTensor())
    transform = transforms.Compose(trans)

    mnistTrain = torchvision.datasets.FashionMNIST('./MNist', train=True, download=True, transform=transform)
    mnistTest = torchvision.datasets.FashionMNIST('./MNist', train=False, download=False, transform=transform)

    trainIter = torch.utils.data.DataLoader(
        mnistTrain, batch_size=batchSize, shuffle=True
    )
    testIter = torch.utils.data.DataLoader(
        mnistTest, batch_size=batchSize, shuffle=True
    )

    return trainIter, testIter

class AlexNet(nn.Module) :
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )
    def forward(self, data) :
        # print('start')
        features = self.conv1(data)
        # print(features.shape)
        # print('1...')
        features = self.conv2(features)
        # print('2...')
        features = self.conv3(features)
        features = self.conv4(features)
        features = self.conv5(features)
        output = self.fc(features.view(data.shape[0], -1))
        return output

def test():
    acc_sum = 0
    batch = 0
    for X,y in testIter:
        X,y = X.cuda(),y.cuda()
        y_hat = net(X)
        print(y_hat.argmax(dim=1))
        acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
        batch += 1
    print('acc_sum %d,batch %d' % (acc_sum,batch))

    return 1.0*acc_sum/(batch*batchSize)
def accuracy(net, test_iter, device) :
    correct_sum = 0.0
    total = 0
    for X, y in test_iter :
        y_predicted = net(X.to(device))
        correct_sum += (y_predicted.argmax(dim=1) == y.to(device)).sum().item()
        total += len(y)
    return correct_sum / total

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 0.001
batchSize = 128
resize = 224
numEpoches = 3
net = AlexNet().to(device)
loss = nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(), lr= lr)

trainIter, testIter = loadData(batchSize, resize)

def train() :
    train_l_sum, batch = 0, 0
    for epoch in range(numEpoches) :
        start = time.time()
        for X, y in trainIter :
            X, y = X.to(device), y.to(device)
            # print(X.shape)
            y_hat = net(X)
            lost = loss(y_hat, y)
            opt.zero_grad()
            lost.backward()
            opt.step()
            train_l_sum += lost.item()
            batch += 1
        testAcc = test()
        end = time.time()
        timePerEpoch = end - start
        print('epoch %d,train_loss %f,test_acc %f,time %f' % 
                (epoch + 1,train_l_sum/(batch*batchSize),testAcc,timePerEpoch))

if os.path.exists('AlexNet.pth') :
    net.load_state_dict(torch.load('AlexNet.pth'))
else :
    train()
    torch.save(net.state_dict(), 'AlexNet.pth')

img = Image.open('4.jpg')
# imgArray = np.array(img)
trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# imgArray = np.transpose(img, (2, 0, 1))[1] / 255.0
imgTensor = trans(img)[0].resize(1, 224, 224)
print(imgTensor.shape)
imgBatch = imgTensor.unsqueeze(0)
print(imgTensor.shape)
# imgTensor = imgTensor.reshape(1, 1, 224, 224).float()
print(imgBatch.shape)
print('predicted : ', net(imgBatch.to(device)).argmax(dim=1))
print(F.softmax(net(imgBatch.to(device))[0], dim=0))

visGraph = hiddenlayer.build_graph(net, imgBatch.to(device))
visGraph.theme = hiddenlayer.graph.THEMES["blue"].copy()
visGraph.save('./AlexNet')