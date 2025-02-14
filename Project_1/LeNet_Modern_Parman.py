import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

#hyper parameters
seed = 1
num_epochs = 20
num_classes = 10
lr = 0.001
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

model = LeNet(lr, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_loss_per_epoch = []
val_loss_per_epoch = []
val_acc_per_epoch = []
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
    ys = []
    preds = []
    val_acc = []
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)

        pred = model(x)
        _, predicted_labels = torch.max(pred, 1)

        ys.append(y.detach().cpu().data.numpy())
        preds.append(predicted_labels.detach().cpu().data.numpy())

        loss = loss_fn(model.forward(x), y)

        val_loss.append(loss.item())
    ys = np.concatenate(ys)
    preds = np.concatenate(preds)
    correct = [1 if j == i else 0 for i, j in zip(ys, preds)]
    total = ys.shape[0]
    val_acc_per_epoch.append(sum(correct) / total)
    train_loss_per_epoch.append(np.mean(train_loss))
    val_loss_per_epoch.append(np.mean(val_loss))
    print("epoch: {}/{}, train loss: {:.3f} | val loss: {:.3f}".format(epoch, num_epochs, np.mean(train_loss),
                                                                       np.mean(val_loss)))

plt.plot(range(1, num_epochs + 1), train_loss_per_epoch, label="training loss")
plt.plot(range(1, num_epochs + 1) ,val_loss_per_epoch, '-', label='validation loss')
plt.title(f"Training and Validation loss across {num_epochs} epochs")
plt.xticks([i for i in range(1, num_epochs+1)])
plt.legend(loc="upper right")
plt.ylim(top=4, bottom=0)
plt.savefig(f"./LeNet_Modern_Figures/Loss_per_{num_epochs}_epochs.png")
plt.show()
plt.close()

plt.plot(range(1, num_epochs + 1), val_acc_per_epoch)
plt.title(f"Validation accuracy across {num_epochs} epochs")
plt.ylim(top=1, bottom=0)
plt.savefig(f"./LeNet_Modern_Figures/Accuracy_per_{num_epochs}_epochs.png")
plt.show()
plt.close()