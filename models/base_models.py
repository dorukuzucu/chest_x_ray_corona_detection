import torch
from torch import nn


def conv2d_block(in_channels,out_channels, bn=False):
    """
    :param in_channels: number of input channels for conv net, int
    :param out_channels: number of output channels for conv net, int
    :param bn: batch normalization flag, boolean. Adds a batch norm layer between conv and Relu if bn is set to True
    :return: Sequential layers, sub-network consists of conv bn relu
    """
    layers = []
    layers.append(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),stride=1,padding=1))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class FeatureExtractor(nn.Module):
    """
    Feature extractor module similar to what is used in VGG16
    """
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv_block_1 = conv2d_block(in_channels=in_channels,out_channels=64,bn=True)
        self.conv_block_2 = conv2d_block(in_channels=64, out_channels=64, bn=True)
        self.max_pool_1 = nn.MaxPool2d(stride=2,kernel_size=2)
        self.conv_block_3 = conv2d_block(in_channels=64, out_channels=128, bn=True)
        self.conv_block_4 = conv2d_block(in_channels=128, out_channels=128, bn=True)
        self.max_pool_2 = nn.MaxPool2d(stride=2, kernel_size=2)
        self.conv_block_5 = conv2d_block(in_channels=128, out_channels=512, bn=True)
        self.conv_block_6 = conv2d_block(in_channels=512, out_channels=512, bn=True)
        self.max_pool_3 = nn.MaxPool2d(stride=2, kernel_size=2)
        self.conv_block_7 = conv2d_block(in_channels=512, out_channels=out_channels, bn=True)
        self.conv_block_8 = conv2d_block(in_channels=out_channels, out_channels=out_channels, bn=True)
        self.max_pool_4 = nn.MaxPool2d(stride=2, kernel_size=2)
        self.conv_layer = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=(3,3),stride=1,padding=1)

    def forward(self,x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.max_pool_1(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.max_pool_2(x)
        x = self.conv_block_5(x)
        x = self.conv_block_6(x)
        x = self.max_pool_3(x)
        x = self.conv_block_7(x)
        x = self.conv_block_8(x)
        x = self.max_pool_4(x)
        x = self.conv_layer(x)
        return x


class FeatureClassifier(nn.Module):
    """
    Takes vector inputs and returns class probabilities
    """
    def __init__(self,in_features,num_classes):
        super().__init__()
        self.fc_1 = nn.Linear(in_features=in_features,out_features=in_features//5)
        self.relu_1 = nn.ReLU()
        self.fc_2 = nn.Linear(in_features=in_features//5,out_features=in_features//5)
        self.relu_2 = nn.ReLU()
        self.fc_3 = nn.Linear(in_features=in_features//5,out_features=num_classes)
        self.s_max = nn.Softmax()

    def forward(self,x):
        x = self.fc_1(x)
        x = self.relu_1(x)
        x = self.fc_2(x)
        x = self.relu_2(x)
        x = self.fc_3(x)
        out = self.s_max(x)
        return out