# %%
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random

# %%
#inputs
# 1. 5d tensor - (batchsz,depth,height,width,channels) 3d images
# 2. ksize - kernel size in 3d conv blocks 

# %%
#Network Blocks
class InitialConvBlock(nn.Module):
    def __init__(self,ksize):
        super(InitialConvBlock,self).__init__()
        self.conv = nn.Conv3d(1,32,kernel_size=ksize,stride=1,padding=tuple(k//2 for k in ksize))
    
    def forward(self,x):
        return F.relu(self.conv(x))

class ConvBlock(nn.Module):
    def __init__(self,ksize):
        super(ConvBlock,self).__init__()
        self.conv = nn.Conv3d(32,32,kernel_size=ksize,stride=1,padding=tuple(k//2 for k in ksize))
    
    def forward(self,x):
        return F.relu(self.conv(x))

class FinalConvBlock(nn.Module):
    def __init__(self,ksize):
        super(FinalConvBlock,self).__init__()
        self.conv1 = nn.Conv3d(32,32,kernel_size=ksize,stride=1,padding=tuple(k//2 for k in ksize))
        self.conv2 = nn.Conv3d(32,1,kernel_size=ksize,stride=1,padding=tuple(k//2 for k in ksize))
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        return self.conv2(x)

# %%
class FloodFillingNetwork(nn.Module):
    def __init__(self,num_modules,ksize):
        super(FloodFillingNetwork,self).__init__()
        assert num_modules > 2 #At least one ConvBlock and one FinalConvBlock
        self.initialconv = InitialConvBlock(ksize)
        self.convblock = nn.ModuleList([ConvBlock(ksize) for i in range(num_modules-1)])
        self.finalconv = FinalConvBlock(ksize)
    
    def forward(self,x):
        x = self.initialconv(x)
        for block in self.convblock:
            x = block(x)
        x = self.finalconv(x)
        return x


