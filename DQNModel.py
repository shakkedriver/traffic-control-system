import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class DQNModel(nn.Module):

    def __init__(self):
        super(DQNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(8, 5), padding='same')
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(8, 5), padding='same')
        self.linear = nn.Linear(500, 3)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()

    def forward(self, state):
        state = self.relu(self.conv1(state))
        state = self.max_pool(state)
        state = self.relu(self.conv2(state))
        state = self.max_pool(state)
        return self.linear(state)
        # x = F.relu(self.bn3(self.conv3(x)))
        # return self.head(x.view(x.size(0), -1))
