##implementing LeNet model in PyTorch from scratch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

#layersP:Convolutional layer, Pooling layer, Fully connected layer
class ConvLayers(nn.Module):
    def __init__(self):
        super(ConvLayers, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1,padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        return x
    
class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(16*5*5,120) #16*5*5 is the number of neurons in the second pooling layer
        self.fc2 = nn.Linear(120,84) #84 is the number of neurons in the second fully connected layer
        self.fc3 = nn.Linear(84,10) #10 is the number of neurons in the output layer

    def forward(self, x):
        x = x.view(-1, 16*5*5) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.net = nn.Sequential(ConvLayers(), FCNN())

    def forward(self, x):
        x = self.net(x)
        return x