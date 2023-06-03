"""
Conditional Generative Adversarial Network (cGAN) model
"""

import torch
from torch.nn import Module, Linear,\
    Conv2d, MaxPool2d, Dropout, Dropout2d,\
    Flatten, Sequential, ELU, Softmax,\
    BatchNorm1d, BatchNorm2d, LeakyReLU,\
    ConvTranspose2d, Sigmoid

class Generator(Module):
    """
    Class to create the generator model
    """
    def __init__(self,
                 input_size,
                 num_classes: int=10,
                 filters:tuple=(128, 64),
                 kernel_size:int=5,
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
        num_classes : int, optional
            Number of classes. The default is 10.
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
            Height of the input image. The default is 7.
        width : int, optional
            Width of the input image. The default is 7.
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
        
        # Save parameters
        self.input_size = input_size
        self.num_classes = num_classes
        self.channels = channels
        self.height = height
        self.width = width

        self.fc_net = Sequential(
            Linear(
                in_features=input_size + num_classes,
                out_features=channels * width * height,
                bias=bias
            ),
            BatchNorm1d(
                num_features=channels * width * height
            ),
            LeakyReLU()
        )

        f1, f2 = filters
        self.conv_model = Sequential(
            ConvTranspose2d(
                in_channels=channels + num_classes,
                out_channels=f1,
                kernel_size=kernel_size,
                bias=bias,
                padding=padding
            ),
            BatchNorm2d(
                num_features=f1
            ),
            LeakyReLU(),
            
            ConvTranspose2d(
                in_channels=f1 + num_classes,
                out_channels=f2,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
                padding=padding,
                output_padding=output_padding
            ),
            BatchNorm2d(
                num_features=f2
            ),
            LeakyReLU(),
            
            ConvTranspose2d(
                in_channels=f2 + num_classes,
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


    def forward(self, x, labels):
        """
        Forward pass
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        labels : torch.Tensor
            Labels tensor
            
        Returns
        -------
        y : torch.Tensor
            Output tensor
        """
        x = torch.cat((x, labels), dim=1)
        y = self.fc_net(x)
        y = y.reshape((
            -1,
            self.channels,
            self.width,
            self.height
        ))
        y = torch.cat((y, y.new_full((y.size(0), y.size(1), y.size(2), y.size(3)), y.size(1) * y.size(2) * y.size(3), dtype=torch.float32).unsqueeze(-1) * y.new_full((y.size(0), y.size(1), y.size(2), y.size(3)), y.size(1) * y.size(2) * y.size(3), dtype=torch.float32).unsqueeze(-1)), dim=1)
        y = torch.cat((y, y.new_full((y.size(0), y.size(1), y.size(2), y.size(3)), y.size(1) * y.size(2) * y.size(3), dtype=torch.float32).unsqueeze(-1) * y.new_full((y.size(0), y.size(1), y.size(2), y.size(3)), y.size(1) * y.size(2) * y.size(3), dtype=torch.float32).unsqueeze(-1)), dim=1)
        y = torch.cat((y, y.new_full((y.size(0), y.size(1), y.size(2), y.size(3)), y.size(1) * y.size(2) * y.size(3), dtype=torch.float32).unsqueeze(-1) * y.new_full((y.size(0), y.size(1), y.size(2), y.size(3)), y.size(1) * y.size(2) * y.size(3), dtype=torch.float32).unsqueeze(-1)), dim=1)
        y = self.conv_model(y)
        return y
        
        #y = torch.cat([x, labels], dim=1)
        #y = self.fc_net(y)
        #y = y.reshape((-1, self.channels, self.height, self.width))
        #y = self.conv_model(y)
        #return y
    
    
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
                 in_channels=1,
                 num_classes=10,
                 filters=(64, 128, 128),
                 kernel_size=5,
                 padding=2,
                 stride=2,
                 dropout=0.3):
        """
        Initialize the discriminator model
        
        Parameters
        ----------
        in_channels : int, optional
            Number of input channels. The default is 1.
        num_classes : int, optional
            Number of classes. The default is 10.
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

        self.model = Sequential(
            Conv2d(in_channels=in_channels + num_classes,
                   out_channels=f1,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding
            ),
            LeakyReLU(),
            Dropout(
                dropout
            ),

            Conv2d(
                in_channels=f1 + num_classes,
                out_channels=f2,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            LeakyReLU(),
            Dropout(
                dropout
            ),

            Conv2d(
                in_channels=f2 + num_classes,
                out_channels=f3,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            LeakyReLU(),
            Dropout(
                dropout
            ),

            Flatten(),
            Linear(
                in_features=f3 * 10 * 5,
                out_features=1
            )
        )

        self.apply(self._init_weights)


    def forward(self,
                x,
                labels):
        """
        Forward pass
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        labels : torch.Tensor
            Labels tensor
        
        Returns
        -------
        y : torch.Tensor
            Output tensor
        """        
        x = torch.cat((x, labels), dim=1)
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
                       real_labels,
                       fake_labels,
                       criterion,
                       device):
    """
    Compute the discriminator loss for a CGAN
    
    Parameters
    ----------
    real_output : torch.Tensor
        Output of the discriminator for the real images
    fake_output : torch.Tensor
        Output of the discriminator for the fake images
    real_labels : torch.Tensor
        Ground truth labels for the real images
    fake_labels : torch.Tensor
        Ground truth labels for the fake images
    criterion : torch.nn.Module
        Loss function
    device : str or torch.device
        Device to perform the computation
        
    Returns
    -------
    loss : torch.Tensor
        Discriminator loss
    """
    real_loss = criterion(real_output, real_labels.to(device))
    fake_loss = criterion(fake_output, fake_labels.to(device))
    loss = real_loss + fake_loss
    return loss


def generator_loss(fake_output, real_labels, criterion, device):
    """
    Compute the generator loss for a CGAN
    
    Parameters
    ----------
    fake_output : torch.Tensor
        Output of the discriminator for the fake images
    real_labels : torch.Tensor
        Ground truth labels for the real images
    criterion : torch.nn.Module
        Loss function
    device : str or torch.device
        Device to perform the computation
        
    Returns
    -------
    loss : torch.Tensor
        Generator loss
    """
    loss = criterion(fake_output, real_labels.to(device))
    return loss
