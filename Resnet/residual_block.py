#Implementing Resnet model from scratch
import torch
import torch.nn as nn
import torch.nn.functional as F

#Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.g = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        #Shortcut connection
        self.shortcut = nn.Sequential()
        if stride!=1 or in_channels!=out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self,x):
        g_x = self.g(x)
        f_x = g_x+self.shortcut(x)   #y=F(x)+x
        return nn.ReLU()(f_x)