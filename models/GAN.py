"""
GAN model
"""

import torch
from torch.nn import Module, Linear,\
                     Conv2d, Dropout, \
                     Flatten, Sequential,\
                     BatchNorm1d, BatchNorm2d,\
                     LeakyReLU, ConvTranspose2d,\
                     Sigmoid, BCEWithLogitsLoss

import time as time


class Generator(Module):
    """
    Class to create the generator model
    """
    def __init__(self,
                 input_size,
                 filters: tuple=(128, 64),
                 kernel_size: int=5,
                 channels:int=256,
                 height:int=10,
                 width:int=20,
                 bias:bool=False,
                 padding:int=2,
                 stride:int=2,
                 output_padding:int=1):
        """
        Constructor
        
        Parameters
        ----------
        input_size : int
            Size of the input vector
        filters : tuple, optional
            Number of filters for each convolutional layer.
            The default is (128, 64).
        kernel_size : int, optional
            Kernel size for the convolutional layers.
            The default is 5.
        channels : int, optional
            Number of channels for the convolutional layers.
            The default is 256.
        height : int, optional
            Height of the input image.
            The default is 20.
        width : int, optional
            Width of the input image.
            The default is 10.
        bias : bool, optional
            Whether to use bias in the convolutional layers.
            The default is False.
        padding : int, optional
            Padding for the convolutional layers.
            The default is 2.
        stride : int, optional
            Stride for the convolutional layers.
            The default is 2.
        output_padding : int, optional
            Output padding for the convolutional layers.
            The default is 1.
        """

        super().__init__()

        # Save the input size
        self.input_size = input_size

        # Save the parameters
        self.channels = channels
        self.height = height
        self.width = width

        self.fc_net = Sequential(
            Linear(
                in_features=input_size,
                out_features=channels*width*height,
                bias=bias
            ),
            BatchNorm1d(
                num_features=channels*width*height
            ),
            LeakyReLU()
        )

        f1, f2 = filters
        self.conv_model = Sequential(
            ConvTranspose2d(
                in_channels=channels,
                out_channels=f1,
                kernel_size=kernel_size,
                bias=bias,
                padding=padding
            ),
            BatchNorm2d(
                num_features=(f1)
            ),
            LeakyReLU(),
            ConvTranspose2d(
                in_channels=f1,
                out_channels=f2,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
                padding=padding,
                output_padding=output_padding
            ),
            BatchNorm2d(
                num_features=(f2)
            ),
            LeakyReLU(),
            ConvTranspose2d(
                in_channels=f2,
                out_channels=1,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
                padding=padding,
                output_padding=output_padding
            ),
            Sigmoid()
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
        y = self.fc_net(x)
        y = y.reshape((
            -1,
            self.channels,
            self.width,
            self.height
        ))
        y = self.conv_model(y)
        return y

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
    def __init__(self,
                 in_channels:int=1,
                 filters: tuple=(64, 128, 128),
                 kernel_size: int=5,
                 padding:int=2,
                 stride:int=2,
                 dropout:float=0.3):
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

        f1, f2, f3 = filters
        
        # Create the model
        self.model = Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=f1,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            LeakyReLU(),
            Dropout(dropout),

            Conv2d(
                in_channels=f1,
                out_channels=f2,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            LeakyReLU(),
            Dropout(dropout),

            Conv2d(
                in_channels=f2,
                out_channels=f3,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            LeakyReLU(),
            Dropout(dropout),

            Flatten(),
            ##### TODO: Check if this is correct
            Linear(f3 * 10 * 5, 1)
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
        y = self.model(x)
        return y


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


######################################################################
# Loss functions
######################################################################
def discriminator_loss(real_output,
                       fake_output,
                       criterion=BCEWithLogitsLoss(),
                       device='cpu'):
    """
    Compute the discriminator loss
    
    Parameters
    ----------
    real_output : torch.Tensor
        Output of the discriminator for the real images
    fake_output : torch.Tensor
        Output of the discriminator for the fake images
    criterion : torch.nn.Module
        Loss function
        
    Returns
    -------
    loss : torch.Tensor
        Discriminator loss
    """
    real_loss = criterion(
        real_output,
        torch.ones_like(
            real_output,
            device=device
        )
    )
    fake_loss = criterion(
        fake_output,
        torch.zeros_like(
            fake_output,
            device=device
        )
    )
    loss = real_loss + fake_loss
    return loss


def generator_loss(fake_output,
                   criterion=BCEWithLogitsLoss(),
                   device='cpu'):
    """
    Compute the generator loss
    
    Parameters
    ----------
    fake_output : torch.Tensor
        Output of the discriminator for the fake images
    criterion : torch.nn.Module
        Loss function
        
    Returns
    -------
    loss : torch.Tensor
        Generator loss
    """
    loss = criterion(
        fake_output,
        torch.ones_like(
            fake_output,
            device=device
        )
    )
    return loss