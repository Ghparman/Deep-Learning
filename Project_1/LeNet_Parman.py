import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

#hyper parameters
seed = 1
num_epochs = 10
num_classes = 10
lr = 0.15
loss_fn = torch.nn.CrossEntropyLoss()
device = torch.device("cpu")

class LeNet(nn.Module):
    def __init__(self,lr,num_classes):
        super().__init__()
        self.Network = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120), nn.Sigmoid(),
            nn.LazyLinear(84), nn.Sigmoid(),
            nn.LazyLinear(num_classes))

    def forward(self, inputs):
        preds = self.Network(inputs)
        return preds

data = pd.read_csv('./data/fashion-mnist_test.csv')

X = data.drop('label', axis=1).values
y = data['label'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.8, random_state=seed)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).reshape(-1, 1, 28, 28)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).reshape(-1, 1, 28, 28)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print("training...")

model = LeNet(lr,num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(1, num_epochs + 1):
    train_loss = []
    val_loss = []
    model.train()

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        loss = loss_fn(model.forward(x), y)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    model.eval()
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)

        loss = loss_fn(model.forward(x), y)

        val_loss.append(loss.item())

    print("epoch: {}/{}, train loss: {:.3f} | val loss: {:.3f}".format(epoch, num_epochs, np.mean(train_loss),
                                                                       np.mean(val_loss)))

