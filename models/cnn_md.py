'''
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

TODO:
    - add the link to the data
    - add the link to the paper
    - add dropout
    - add initial weights

'''

import torch
from torch.nn import Module, Linear,\
                     Conv2d, MaxPool2d,\
                     Dropout,\
                     Flatten, Sequential,\
                     ELU, Softmax

class cnn_md(Module):
    '''
    Class to create the model for the mDoppler data
    '''

    def __init__(self,
                 in_channels: int=1,
                 filters: tuple=(8, 16, 32, 64),
                 kernel_size: tuple=(3, 3),
                 stride: int=1,
                 pool_size: tuple=(2, 2),
                 pool_stride: int=2,
                 padding= 'same',
                 dilation: int=1,
                 groups: int=1,
                 bias: bool=True,
                 padding_mode: str='zeros'):
        '''
        Constructor
        '''
        super().__init__()
        f1, f2, f3, f4 = filters

        # Convolutional layers
        self.cnn = Sequential(

            Conv2d(in_channels=in_channels, out_channels=f1, kernel_size=kernel_size, stride=stride, padding=padding),
            ELU(),
            MaxPool2d(kernel_size=pool_size, stride=pool_stride),

            Conv2d(in_channels=f1, out_channels=f2, kernel_size=kernel_size, stride=stride, padding=padding),
            ELU(),
            MaxPool2d(kernel_size=pool_size, stride=pool_stride),

            Conv2d(in_channels=f2, out_channels=f3, kernel_size=kernel_size, stride=stride, padding=padding),
            ELU(),
            MaxPool2d(kernel_size=pool_size, stride=pool_stride),

            Conv2d(in_channels=f3, out_channels=f4, kernel_size=kernel_size, stride=stride, padding=padding),
            ELU(),
            MaxPool2d(kernel_size=pool_size, stride=pool_stride)
        )

        # Flatten the output of the convolutional layers
        self.flatten = Flatten()

        # Fully connected layers
        self.fc = Sequential(
            # num_filters * height * width
            # height = (input_height - kernel_size + 2 * padding) / stride + 1
            # width = (input_width - kernel_size + 2 * padding) / stride + 1
            Linear(in_features=f4*6*2, out_features=128), ### 6 and 2 are the height and width of the input
            ELU(), # not sure if this is the right activation function
            #Dropout(p=0.2),
            Linear(in_features=128, out_features=6),
            Linear(in_features=6, out_features=14),
            Softmax(dim=1)
        )

        self.apply(self._init_weights)


    def _init_weights(self, module):
        '''
        Initialize the weights of the model
        '''
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()


    def forward(self, x):
        '''
        Forward pass
        '''
        # Convolutional layers
        x = self.cnn(x)
        # Flatten the output of the convolutional layers
        x = self.flatten(x)        
        # Fully connected layers
        x = self.fc(x)
        return x

    def save(self, path: str):
        '''
        Save the model
        '''
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        '''
        Load the model
        '''
        self.load_state_dict(torch.load(path))
