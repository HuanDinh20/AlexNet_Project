from torch import nn
import torch


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(),
            nn.Linear(4096, 10))
        self.net.apply(self.init_xavier_uniform)

    def forward(self, X):
        X = self.net(X)
        return X

    def layer_summary(self, X_shape: tuple):
        X = torch.rand(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, "output shape: ", X.shape)

    @staticmethod
    def init_xavier_uniform(layer):
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.0001)
