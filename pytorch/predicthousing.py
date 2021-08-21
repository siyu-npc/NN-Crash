import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

housedata = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(housedata.data, housedata.target,
    test_size=0.3, random_state=123)

scale = StandardScaler()
X_train_s = scale.fit_transform(X_train)
X_test_s = scale.transform(X_test)

housedatadf = pd.DataFrame(data=X_train_s, columns=housedata.feature_names)
housedatadf['target'] = y_train
datacor = np.corrcoef(housedatadf.values, rowvar=0)
datacor = pd.DataFrame(data=datacor, columns=housedatadf.columns, index=housedatadf.columns)
# plt.figure(figsize=(8, 6))
# ax = sns.heatmap(datacor, square=True, annot=True, fmt = ".3f", linewidths=0.5, 
#     cmap="YlGnBu", cbar_kws={"fraction":0.046, "pad":0.03})
# plt.show()

train_xt = torch.from_numpy(X_train_s.astype(np.float32))
train_yt = torch.from_numpy(y_train.astype(np.float32))
test_xt = torch.from_numpy(X_test_s.astype(np.float32))
test_yt = torch.from_numpy(y_test.astype(np.float32))
train_data = Data.TensorDataset(train_xt, train_yt)
test_data = Data.TensorDataset(test_xt, test_yt)
train_loader = Data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=1)

class MLPregression(nn.Module) :
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(8, 100, bias=True)
        self.hidden2 = nn.Linear(100, 100)
        self.hidden3 = nn.Linear(100, 50)
        self.predict = nn.Linear(50, 1)

    def forward(self, x) :
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        output = self.predict(x)
        return output

if __name__ == '__main__' :
    mlpreg = MLPregression()
    optimizer = torch.optim.SGD(mlpreg.parameters(),lr=0.01)
    loss_func = nn.MSELoss()
    train_loss_all = []

    for epoch in range(30) :
        train_loss = 0
        train_num = 0
        for step, (b_x, b_y) in enumerate(train_loader) :
            output = mlpreg(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_x.size(0)
            train_num += b_x.size(0)
        train_loss_all.append(train_loss / train_num)

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_all, "ro-", label='Train Loss')
    plt.legend()
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.show()