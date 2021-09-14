import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, row, col):
        super().__init__()
        self.r = row
        self.c = col

        # feature extract layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)

        # action policy layers
        self.act_conv1 = nn.Conv2d(64, 4, kernel_size=(3, 3), padding=1)
        self.act_conv2 = nn.Conv2d(4, 1, kernel_size=(3, 3), padding=1)

        # state value layers
        self.val_conv1 = nn.Conv2d(64, 4, kernel_size=(3, 3), padding=1)
        self.val_conv2 = nn.Conv2d(4, 1, kernel_size=(row, col))

        # function layer
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        # state: (N, 3, row, col)
        x = self.relu(self.conv1(state))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # action head
        act = self.relu(self.act_conv1(x))
        act = self.relu(self.act_conv2(act))
        act = self.softmax(act.view(-1, self.r * self.c)).view(-1, self.r, self.c)

        # value head
        value = self.relu(self.val_conv1(x))
        value = self.relu(self.val_conv2(value))
        value = self.sig(value.view(-1, 1))

        return act, value



