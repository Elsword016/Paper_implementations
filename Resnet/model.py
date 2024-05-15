#Implementing Resnet model from scratch
import torch
import torch.nn as nn
import torch.nn.functional as F
from residual_block import ResidualBlock

#intital layer
class ResNet(nn.Module):
    def __init__(self,ResidualBlock,num_classes=10):
        super(ResNet,self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(ResidualBlock, 64, 2)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes) #fully connected layer
    
    def make_layer(self,block,out_ch,blocks,stride=1):
        layers = []
        layers.append(block(self.in_channels,out_ch,stride))
        self.in_channels = out_ch
        for _ in range(1,blocks):
            layers.append(block(out_ch,out_ch))
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x 