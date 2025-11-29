import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        
        # first sequence: Conv2D - BatchNorm - ReLU
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # second sequence: Conv2D - BatchNorm
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # skip connection adaptation
        self.skip_connection = None
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # final relu
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # store input for skip connection
        identity = x
        
        # first sequence: Conv - BN - ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        # second sequence: Conv - BN
        out = self.conv2(out)
        out = self.bn2(out)
        
        # adapt skip connection if needed
        if self.skip_connection is not None:
            identity = self.skip_connection(x)
        
        # add skip connection and apply final relu
        out += identity
        out = self.relu2(out)
        
        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        
        # initial convolution block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # resnet blocks
        self.resblock1 = ResBlock(64, 64, stride=1)
        self.resblock2 = ResBlock(64, 128, stride=2)
        self.resblock3 = ResBlock(128, 256, stride=2)
        self.resblock4 = ResBlock(256, 512, stride=2)
        
        # global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # flatten layer (handled in forward)
        
        # fully connected layer
        self.fc = nn.Linear(512, 2)
        
        # sigmoid activation
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # initial convolution block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # resnet blocks
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        
        # global average pooling
        x = self.global_avg_pool(x)
        
        # flatten
        x = torch.flatten(x, 1)
        
        # fully connected layer
        x = self.fc(x)
        
        # sigmoid activation
        x = self.sigmoid(x)
        
        return x
