#! /usr/bin/env python

# reference : https://github.com/Zhenye-Na/e2e-learning-self-driving-cars

import torch.nn as nn
import torch.nn.functional as F

class End2End(nn.Module):

    def __init__(self):
        """
        Image normalization to avoid saturation and make gradients work better.
        Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
        Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
        Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
        Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
        Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
        Drop out (0.5)
        Fully connected: neurons: 100, activation: ELU
        Fully connected: neurons: 50, activation: ELU
        Fully connected: neurons: 10, activation: ELU
        Fully connected: neurons: 1 (output)
        """
        super(End2End, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.Dropout(0.5)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=64*1*18, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.ELU(),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, input):
        output = self.conv_layers(input)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output