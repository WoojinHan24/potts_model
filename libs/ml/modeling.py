from torch import nn


class ClassificationCNN(nn.Module):
    def __init__(self, Q: int, Lx: int, Ly: int):
        super().__init__()
        self.Q = Q
        # the implementation matches https://www.sciencedirect.com/science/article/abs/pii/S0003491618300459
        self.conv_in = nn.Conv2d(Q, Q, 3, 1, 1, padding_mode="zeros")
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(2)
        self.fc_out = nn.Linear(Q * (Lx // 2) * (Ly // 2), Q)

    def forward(self, x):
        # assume 4D input [N, Q, L, L]: 2D lattice batched in one-hot
        x = self.pool(self.relu(self.conv_in(x))).reshape(x.shape[0], -1)
        x = self.fc_out(x)
        return x

