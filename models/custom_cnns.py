"""
cnn_md.py

This file contains the Model class, which is used to create the model 
for the mDoppler data.

The model is based on the following paper:
[LINK]

The model is composed of 4 convolutional layers, 2 fully connected layers
and 1 output layer. The activation function used is ELU. The output layer
uses the Softmax activation function.

The model is trained using the Adam optimizer and the Cross Entropy loss.

The model is trained on the following data:
[LINK]
"""

import torch
from torch.nn import Module, Linear,\
                     Conv2d, MaxPool2d,\
                     Dropout, Dropout2d,\
                     Flatten, Sequential,\
                     ELU, Softmax, ReLU,\
                     ModuleList, BatchNorm2d


class ConvBlock(Module):
    """
    Class to create a convolutional block
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride: int,
        padding: int,
        pool_kernel_size: int,
        pool_stride: int,
        dropout: float=0.5
    ):
        """
        Make a convolutional layer
        
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : tuple
            Kernel size
        stride : int
            Stride
        padding : int
            Padding
        dropout : float, optional
            Dropout, by default 0.5
            
        Returns
        -------
        Sequential
            Sequential layer
        """
        super().__init__()
        self.conv = Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            ELU(),
            MaxPool2d(
                kernel_size=pool_kernel_size,
                stride=pool_stride
            ),
            Dropout2d(
                p=dropout
            )
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        return self.conv(x)
    
  
class LinearBlock(Module):
    """
    Class to create a linear block
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float=0.5
    ):
        """
        Make a linear layer
        
        Parameters
        ----------
        in_features : int
            Number of input features
        out_features : int
            Number of output features
        dropout : float, optional
            Dropout, by default 0.5
            
        Returns
        -------
        Sequential
            Sequential layer
        """
        super().__init__()
        self.linear = Sequential(
            Linear(
                in_features=in_features,
                out_features=out_features
            ),
            ELU(),
            Dropout(
                p=dropout
            )
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, x):
        return self.linear(x)


class CNNCustom(Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=10,
        filters=(8, 16, 32, 64),
        neurons=(256, 128),
        kernel_size=(3, 3),
        stride=1,
        padding='same',
        pool_size=(2, 2),
        pool_stride=2,
        dropout=(0.2, 0.2, 0.2, 0.2, 0.2, 0.2),
    ):
        super().__init__()

        if len(dropout) != len(filters) + len(neurons):
            raise ValueError(
                "The number of dropout values should be equal to the number of "
                "convolutional layers + the number of linear layers"
            )

        # Convolutional layers
        self.conv_blocks = ModuleList()
        for i, num_filters in enumerate(filters):
            self.conv_blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    pool_kernel_size=pool_size,
                    pool_stride=pool_stride,
                    dropout=dropout[i],
                )
            )
            in_channels = num_filters

        # Linear layers
        self.linear_blocks = ModuleList()
        in_features = filters[-1] * (80 // (2 ** len(filters))) * (40 // (2 ** len(filters)))
        for i, num_neurons in enumerate(neurons):
            self.linear_blocks.append(
                LinearBlock(
                    in_features=in_features,
                    out_features=num_neurons,
                    dropout=dropout[i + len(filters)],
                )
            )
            in_features = num_neurons

        # Output layer
        self.output_layer = Linear(in_features, out_channels)
        self.softmax = Softmax(dim=1)

    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        x = x.view(x.size(0), -1)

        for linear_block in self.linear_blocks:
            x = linear_block(x)

        x = self.output_layer(x)
        x = self.softmax(x)

        return x


    def _init_weights(self, module):
        """
        Initialize the weights of the model
        """
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()


    def save(self, path: str):
        """
        Save the model
        """
        torch.save(self.state_dict(), path)


    def load(self, path: str):
        """
        Load the model
        """
        self.load_state_dict(torch.load(path))