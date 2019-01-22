import torch
import torch.nn as nn
import pandas as pd
import numpy as np


# haven't picked dataset, but I think will use MNIST fashion dataset

# MODEL
class FirstCNN(nn.module):
    def __init__(self):
        super(FirstCNN, self).__init__()
        # 3 channels R-G-B
        self.conv1 = nn.Conv2d(channels_in=3, channels_out=6, kernel=5, stride=1)
        self.relu1 = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel=2)

        self.conv2 = nn.Conv2d(channels_in=6, channels_out=16, kernel=5, stride=1)
        self.relu2 = nn.ReLU()

        self.fullyC = nn.Linear(features_in=16*5*5, features_out=10)

        # technically 4 layers - convolution1, max-pooling, convolution2, fully connected
    def forward(self, input):
        out = self.conv1(input)
        out = self.relu1(out)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.relu(out)

        out = out.view(-1, 16*5*5)  # flattening in order to use Linear fully connected layer
        out = self.fullyC(out)

        return out


model = FirstCNN()






