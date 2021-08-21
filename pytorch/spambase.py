import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import hiddenlayer as hl
from torchviz import make_dot
import numpy as np
import torch.utils.data as Data
from torch.optim import SGD,Adam
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import hiddenlayer as hl

spam = pd.read_csv('C:\\Users\\思雨\\Downloads\\spambase\\spambase.csv')

X = spam.iloc[:,0:57].values
y = spam.label.values

# 将数据随机切分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=123)

# 数据标准化
scales = MinMaxScaler(feature_range=(0,1))
X_train_s = scales.fit_transform(X_train)
x_test_s = scales.transform(X_test)
print('scales=',X_train_s)

# 画图
colname = spam.columns.values[:-1]
# plt.figure(figsize=(20,14))
# for ii in range(len(colname)):
#     plt.subplot(7,9,ii+1)
#     sns.boxplot(x = y_train, y = X_train_s[:,ii])
#     plt.title(colname[ii])
# plt.subplots_adjust(hspace=0.4)
# plt.show()

class MLPclassifica(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(
                in_features=57,
                out_features=30,
                bias=True
            ),
            nn.ReLU()
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(30,10),
            nn.ReLU()
        )
        self.classifica = nn.Sequential(
            nn.Linear(10,2),
            nn.Sigmoid()
        )
    def forward(self,x):
        fc1 = self.hidden1(x)
        fc2 = self.hidden2(fc1)
        output = self.classifica(fc2)
        return fc1,fc2,output

mlpc = MLPclassifica()
x = torch.randn(1,57).requires_grad_(True)
y = mlpc(x)

# 输出网络图
Mymlpcvis = make_dot(y, params=dict(list(mlpc.named_parameters()) + [('x', x)]))
Mymlpcvis.render('spambase',view=False)

# 构造数据集
X_train_nots = torch.from_numpy(X_train.astype(np.float32))
y_train_t = torch.from_numpy(y_train.astype(np.int64))
X_test_nots = torch.from_numpy(X_test.astype(np.float32))
y_test_t = torch.from_numpy(y_test.astype(np.int64))
train_data_nots = Data.TensorDataset(X_train_nots, y_train_t)
train_nots_loader = Data.DataLoader(
    dataset = train_data_nots,
    batch_size = 64,
    shuffle = True,
    num_workers = 1
)

# 训练
optimizer = torch.optim.Adam(mlpc.parameters(),lr = 0.01)
loss_func = nn.CrossEntropyLoss()
history1 = hl.History()
canvas1 = hl.Canvas()
print_step = 25
for epoch in range(15):
    for step, (b_x, b_y) in enumerate(train_nots_loader):
        _,_,output = mlpc(b_x)
        train_loss = loss_func(output, b_y)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        niter = epoch*len(train_nots_loader)+step+1
        if niter % print_step == 0:
            _,_,output = mlpc(X_test_nots)
            _,pre_lab = torch.max(output,1)
            test_accuracy = accuracy_score(y_test_t, pre_lab)
            history1.log(niter, train_loss=train_loss,test_accuracy=test_accuracy)
            with canvas1:
                canvas1.draw_plot(history1['train_loss'])
                canvas1.draw_plot(history1['test_accuracy'])

# 数据标准化后训练
X_train_t = torch.from_numpy(X_train_s.astype(np.float32))
y_train_t = torch.from_numpy(y_train.astype(np.int64))
X_test_t = torch.from_numpy(x_test_s.astype(np.float32))
y_test_t = torch.from_numpy(y_test.astype(np.int64))
train_data = Data.TensorDataset(X_train_t, y_train_t)
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=64,
    shuffle=True,
    num_workers=1
)

optimizer = torch.optim.Adam(mlpc.parameters(),lr = 0.01)
loss_func = nn.CrossEntropyLoss()
history1 = hl.History()
canvas1 = hl.Canvas()
print_step = 25

for epoch in range(15):
    for step, (b_x, b_y) in enumerate(train_loader):
        _,_,output = mlpc(b_x)
        train_loss = loss_func(output, b_y)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        niter = epoch*len(train_loader)+step+1
        if niter % print_step == 0:
            _,_,output = mlpc(X_test_t)
            _,pre_lab = torch.max(output,1)
            test_accuracy = accuracy_score(y_test_t, pre_lab)
            history1.log(niter, train_loss=train_loss,test_accuracy=test_accuracy)
            with canvas1:
                canvas1.draw_plot(history1['train_loss'])
                canvas1.draw_plot(history1['test_accuracy'])