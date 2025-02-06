
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import matplotlib.pyplot as plt
class FashionMNIST(d2l.DataModule):  #@save
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True)

    def text_labels(self, indices):
        """Return text labels."""
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [labels[int(i)] for i in indices]

    def get_dataloader(self, train):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,num_workers=self.num_workers)

    def visualize(self, batch, nrows=1, ncols=8, labels=[]):
        X, y = batch
        if not labels:
            labels = self.text_labels(y)
        d2l.show_images(X.squeeze(1), nrows, ncols, titles=labels)

    class Classifier(d2l.Module):  # @save
        """The base class of classification models."""
        def validation_step(self, batch):
            Y_hat = self(*batch[:-1])
            self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
            self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=self.lr)

        def accuracy(self, Y_hat, Y, averaged=True):
            """Compute the number of correct predictions."""
            Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
            preds = Y_hat.argmax(axis=1).type(Y.dtype)
            compare = (preds == Y.reshape(-1)).type(torch.float32)
            return compare.mean() if averaged else compare

        def loss(self, Y_hat, Y, averaged=True):
            Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
            Y = Y.reshape((-1,))
            return F.cross_entropy(
                Y_hat, Y, reduction='mean' if averaged else 'none')

class SoftmaxRegression(d2l.Classifier):  #@save
    """The softmax regression model."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.LazyLinear(num_outputs))
    def forward(self, X):
        return self.net(X)


data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegression(num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
plt.show()
