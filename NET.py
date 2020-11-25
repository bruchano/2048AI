import torch
import torchvision
import numpy as np
from collections import namedtuple
from PIL import ImageGrab


MOVE = ["UP", "DOWN", "LEFT", "RIGHT"]


class AutoPlayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=10, kernel_size=10, stride=10)
        self.in1 = torch.nn.InstanceNorm2d(num_features=10)

        self.conv2 = torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=4, stride=4)
        self.in2 = torch.nn.InstanceNorm2d(num_features=20)

        self.conv3 = torch.nn.Conv2d(in_channels=20, out_channels=10, kernel_size=3, stride=3)
        self.in3 = torch.nn.InstanceNorm2d(num_features=10)

        self.linear = torch.nn.Linear(160, 4)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.in1(self.conv1(x)))
        x = self.relu(self.in2(self.conv2(x)))
        x = self.relu(self.in3(self.conv3(x)))
        x = x.view(1, -1)
        return self.linear(x)

