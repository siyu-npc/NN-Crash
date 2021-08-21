import torch
import numpy as np
from torch.utils.data import dataloader
from torchvision import datasets
from torchvision.transforms.transforms import Lambda, ToTensor

data = [[1,2],[3,4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

if torch.cuda.is_available() :
    x_np = x_np.to('cuda')
print(x_np.device)

tensor = torch.ones(4,4)
print('First row:', tensor[0])
print('First column:', tensor[:, 0])
tensor[:,1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

y1 = x_data @ x_data.T
y2 = x_data.matmul(x_data.T)
print(y1 == y2)

print(x_data * x_data)

ds = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(
        64, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
)
dataloader = dataloader.DataLoader(ds, batch_size=64,shuffle=True)
features, labels = next(iter(dataloader))
print(labels)
# torch.zeros(3, dtype=torch.float).scatter_(dim=0,index=torch.tensor(y),value=1)
x = torch.tensor([[1.,-1.],[1.,1.]], requires_grad=True)
out = x.pow(2).sum()
print('out=',out)
out.backward()
print(x.grad)
print(x.sign())

a = torch.randn(4,4)
print(a)
print(torch.argmax(a))
print(torch.amax(a, 0))
print(torch.amax(a, 1))
print(a.dim())

X = torch.tensor([[0,0],[0,1],[1,0],[1,1]])
W = torch.tensor([[1,1],[1,1]])
C = torch.tensor([0,-1])
w = torch.tensor([[1,-2]]).T
gx = X.mm(W) + C
gx = torch.where(gx > 0,gx, 0)
print(gx)
print(gx.mm(w))
