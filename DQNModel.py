import torch
# import torchvision
import torch.nn as nn
# import torch.nn.functional as F


class DQNModel(nn.Module):

    def __init__(self):
        super(DQNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding='same')

        self.linear1 = nn.Linear(1600, 300)
        self.linear2 = nn.Linear(300, 3)
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.lrelu = nn.LeakyReLU()

    def forward(self, state):
        state = self.lrelu(self.conv1(state))
        state = self.max_pool(state)
        state = self.lrelu(self.conv2(state))
        state = self.max_pool(state)
        state = self.lrelu(self.linear1(torch.flatten(state, 1)))
        return self.linear2(state)
