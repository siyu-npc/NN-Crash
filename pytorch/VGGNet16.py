import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU
import torchvision
import torchvision.transforms as transforms
import time

class VggNet16(nn.Module) :
    def __init__(self, num_classes = 10):
        super().__init__()
        layers = []
        in_channels = 1
        out_channels = 64

        for i in range(13) :
            layers += [nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.ReLU(inplace=True)]
            in_channels = out_channels

            if i in (1, 3, 6, 9, 12) :
                layers += [nn.MaxPool2d(2, 2)]
                if i != 9 :
                    out_channels *= 2
        self.conv = nn.Sequential(*layers)

        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
    def forward(self, data) :
        x = self.conv(data)
        x = x.view(data.shape[0], -1)
        return x

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

def accuracy(net, test_iter, device) :
    correct_sum = 0.0
    total = 0
    for X, y in test_iter :
        y_predicted = net(X.to(device))
        correct_sum += (y_predicted == y.to(device)).sum().item()
        total += len(y)
    return correct_sum / total
def test():
    acc_sum = 0
    batch = 0
    for X,y in test_iter:
        X,y = X.cuda(),y.cuda()
        y_hat = vggnet16(X)
        acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
        batch += 1
    print('acc_sum %d,batch %d' % (acc_sum,batch))

    return 1.0*acc_sum/(batch*batch_size)
def train(net, data_iter, num_epoches, loss_func, batch_size, lr, device) :
    opt = torch.optim.SGD(net.parameters(), lr=lr)    
    for epoch in range(num_epoches) :
        train_loss_sum, batch, acc_sum = 0, 0, 0
        start_time = time.time()
        for X,y in data_iter :
            torch.cuda.empty_cache()
            X = X.to(device)
            y = y.to(device)
            y_predicted = net(X)
            loss = loss_func(y_predicted, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # print('current loss.item() = ', loss.item())
            train_loss_sum += loss.item()
            batch += 1
        acc_sum = test()
        time_per_epoch = time.time() - start_time
        print('epoch %d,train_loss %f,acc_sun %f,time %f' % 
                (epoch + 1,train_loss_sum/(batch*batch_size),acc_sum,time_per_epoch))

if __name__ == '__main__' :
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16
    train_iter, test_iter = loadData(batch_size, 224)
    lr = 0.001
    torch.cuda.empty_cache()

    vggnet16 = VggNet16().to(device)
    loss_func = nn.CrossEntropyLoss()
    train(vggnet16, train_iter, 3, loss_func, batch_size, lr, device)



