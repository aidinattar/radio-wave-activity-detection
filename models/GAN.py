"""
GAN model
"""

import torch
from torch.nn import Module, Linear,\
                     Conv2d, Dropout, \
                     Flatten, Sequential,\
                     BatchNorm1d, BatchNorm2d,\
                     LeakyReLU, ConvTranspose2d,\
                     Sigmoid, BCEWithLogitsLoss,\
                     Tanh, BCELoss
import numpy as np
import time as time


map_shape = (1, 80, 40)

class Generator(Module):
    """
    Class to create the generator model
    """
    def __init__(self):

        super().__init__()

        
        
        def block(in_feat, out_feat, normalize=True):
            layers = [Linear(in_feat, out_feat)]
            if normalize:
                layers.append(BatchNorm1d(out_feat, 0.8))
            layers.append(LeakyReLU(0.2, inplace=True))
            return layers

        self.model = Sequential(
            *block(100, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            Linear(1024, int(np.prod(map_shape))),
            Tanh()
        )

        self.apply(self._init_weights)

    def forward(self, x):
        """
        Forward pass
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
            
        Returns
        -------
        y : torch.Tensor
            Output tensor
        """
        img = self.model(x)
        img = img.view(img.size(0), *map_shape)
        return img


    def _init_weights(self, module):
        """
        Initialize the weights of the model
        
        Parameters
        ----------
        module : torch.nn.Module
            Module to initialize
        """
        classname = module.__class__.__name__
        if classname.find('Conv') != -1:
            #torch.nn.init.normal_(module.weight.data, 0.0, 0.02)
            torch.nn.init.xavier_uniform_(module.weight.data)
        if classname.find('Linear') != -1:
            #torch.nn.init.normal_(module.weight.data, 0.0, 0.02)
            torch.nn.init.xavier_uniform_(module.weight.data)
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(module.weight.data, 1.0, 0.2)
            torch.nn.init.constant_(module.bias.data, 0)


class Discriminator(Module):
    """
    Class to create the discriminator model
    """
    def __init__(self,):
        """
        Constructor
        
        Parameters
        ----------
        in_channels : int, optional
            Number of input channels. The default is 1.
        filters : tuple, optional
            Number of filters for each convolutional layer.
            The default is (64, 128, 2048).
        kernel_size : int, optional
            Kernel size for the convolutional layers.
            The default is 5.
        padding : int, optional
            Padding for the convolutional layers.
            The default is 2.
        stride : int, optional
            Stride for the convolutional layers.
            The default is 2.
        dropout : float, optional
            Dropout probability. The default is 0.3.
        """

        super().__init__()

        self.model = Sequential(
            Linear(int(np.prod(map_shape)), 512),
            LeakyReLU(0.2, inplace=True),
            Linear(512, 256),
            LeakyReLU(0.2, inplace=True),
            Linear(256, 1),
            Sigmoid(),
        )            
        
        self.apply(self._init_weights)


    def forward(self, img):
        """
        Forward pass
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
            
        Returns
        -------
        y : torch.Tensor
            Output tensor
        """
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


    def _init_weights(self, module):
        """
        Initialize the weights of the model
        
        Parameters
        ----------
        module : torch.nn.Module
            Module to initialize
        """
        classname = module.__class__.__name__
        if classname.find('Conv') != -1:
            #torch.nn.init.normal_(module.weight.data, 0.0, 0.02)
            torch.nn.init.xavier_uniform_(module.weight.data)
        if classname.find('Linear') != -1:
            #torch.nn.init.normal_(module.weight.data, 0.0, 0.02)
            torch.nn.init.xavier_uniform_(module.weight.data)
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(module.weight.data, 1.0, 0.2)
            torch.nn.init.constant_(module.bias.data, 0)