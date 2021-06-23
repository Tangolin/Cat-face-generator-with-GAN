import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.conv import Conv2d, ConvTranspose2d
from torch.nn.modules.linear import Linear
from torchvision import datasets

'''
This generator is based on the implementations in 'Unsupervised Representation Learning 
with Deep Convolutional Generative Adversarial Networks'
'''

class conv_block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, type = ['generator', 'discriminator'], last = False):
        super().__init__()
        self.modules = []
        if type == 'generator':
            self.conv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride=stride, \
                                        padding=padding, bias=False)
            self.modules.append(self.conv)

            if last:
                self.act = nn.Tanh()
                self.modules.append(self.act)
            else:
                self.bn = nn.BatchNorm2d(out_channel)
                self.act = nn.ReLU()
                self.modules.extend([self.bn, self.act])

        if type == 'discriminator':
            self.conv = Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, \
                               padding=padding, bias=False)
            self.modules.append(self.conv)
            
            if last:
                self.act = nn.Sigmoid()
                self.modules.append(self.act)
            else:
                self.bn = nn.BatchNorm2d(out_channel)
                self.act = nn.LeakyReLU(0.2, inplace=True)
                self.modules.extend([self.bn, self.act])
        
        nn.init.normal_(self.conv.weight, mean=0, std=0.02)
        self.block = nn.Sequential(*self.modules)
    
    def forward(self, input):

        return self.block(input)


class generator_net(nn.Module):
    def __init__(self, in_channels, out_channels, feature_maps = 64):
        super().__init__()
        self.blk1 = conv_block(in_channels, feature_maps * 16, kernel_size = 4, \
                               stride=1, padding=0, type = 'generator')
        # outputs = (N, C, 4, 4)

        self.blk2 = conv_block(feature_maps * 16, feature_maps * 8, kernel_size=4, \
                               stride=2, padding=1, type = 'generator')
        # outputs = (N, C, 8, 8)

        self.blk3 = conv_block(feature_maps * 8 , feature_maps * 4, kernel_size=4, \
                               stride=2, padding=1, type='generator')
        # outputs = (N, C, 16, 16)
        
        self.blk4 = conv_block(feature_maps * 4 , feature_maps * 2, kernel_size=4, \
                               stride=2, padding=1, type='generator')
        #outputs = (N, C, 32, 32)

        self.blk5 = conv_block(feature_maps * 2 , out_channels, kernel_size=4, \
                               stride=2, padding=1, type='generator', last=True)
        #outputs = (N, 3, 64, 64)

        self.net = nn.Sequential(
            self.blk1,
            self.blk2,
            self.blk3,
            self.blk4,
            self.blk5
        )

    def forward(self, input):

        return self.net(input)
        

class discriminator_net(nn.Module):
    def __init__(self, in_channel, feature_maps):
        super().__init__()
        self.blk1 = conv_block(in_channel, feature_maps, kernel_size=4, \
                               stride=2, padding=1, type='discriminator')
        # (N, C, 32, 32)

        self.blk2 = conv_block(feature_maps, feature_maps * 2, kernel_size=4, \
                               stride=2, padding=1, type='discriminator')
        # (N, C, 16, 16)

        self.blk3 = conv_block(feature_maps * 2, feature_maps * 4, kernel_size=4, \
                               stride=2, padding=1, type='discriminator')
        # (N, C, 8, 8)

        self.blk4 = conv_block(feature_maps * 4, 1, kernel_size=8, stride=1, padding=0, \
                               type='discriminator', last=True)

        # self.blk4 = conv_block(feature_maps * 4, feature_maps * 8, kernel_size=4, \
        #                        stride=1, padding=1, type='discriminator',)

        # self.blk5 = conv_block(feature_maps * 8, 1, kernel_size = 4, stride=1, padding=1, \
        #                        type='discriminator', last=True)

        self.net = nn.Sequential(
            self.blk1,
            self.blk2, 
            self.blk3,
            self.blk4
            # self.blk5
        )

    def forward(self, input):

        return self.net(input)


class autoencoder(nn.Module):
    def __init__(self, height, width, feature_vector_length):
        # init_shape = (N, C, hx, hy)
        super().__init__()
        self.height = height
        self.width = width
        self.net = nn.Sequential(
            nn.Linear(height * width, 256),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True),
            nn.Linear(128, feature_vector_length),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        assert input.shape == (self.height, self.width), 'Wrong input image size'
        input = input.flatten()

        return self.net(input)