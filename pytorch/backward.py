import torch
from torch.autograd import Variable

x = Variable(torch.randn(3), requires_grad=True)
y = Variable(torch.randn(3), requires_grad=True)
z = Variable(torch.randn(3), requires_grad=True)

t = x + y
print('t = ',t)
l = t.dot(z)
print('l = ',l)
l.backward(retain_graph=True)
print(x.grad)
print(y.grad)
print(z.grad)

x.grad.data.zero_()
y.grad.data.zero_()
z.grad.data.zero_()
t.backward(z)
print(x.grad)
print(y.grad)


