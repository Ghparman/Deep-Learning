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
num_epochs = 5
num_classes = 10
lr = 0.1
loss_fn = torch.nn.CrossEntropyLoss()
batch_size = 32
device = torch.device("cpu")

class LeNet(nn.Module):
    def __init__(self,lr,num_classes):
        super().__init__()
        self.Network = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120), nn.ReLU(),
            nn.LazyLinear(84), nn.ReLU(),
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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("training...")

model = LeNet(lr,num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_loss_per_epoch = []
val_loss_per_epoch = []
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
    train_loss_per_epoch.append(np.mean(train_loss))
    val_loss_per_epoch.append(np.mean(val_loss))
    print("epoch: {}/{}, train loss: {:.3f} | val loss: {:.3f}".format(epoch, num_epochs, np.mean(train_loss),
                                                                       np.mean(val_loss)))
fig, axs = plt.subplots(2)
fig.suptitle(f"Training and Validation loss across {num_epochs} epochs")
axs[0].plot(train_loss_per_epoch)
axs[0].set_ylabel("training loss")
axs[0].set_xticks([i for i in range(1, num_epochs+1)])
axs[0].spines['bottom'].set_position(('data', 1))
axs[1].plot(val_loss_per_epoch)
axs[1].set_ylabel("validation loss")
axs[1].set_xticks([i for i in range(1,num_epochs+1)])
axs[1].spines['bottom'].set_position(('data', 1))


plt.show()
