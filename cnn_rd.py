'''
cnn_rd.py

This file contains the Model class, which is used to create the model 
for the rdn data.

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

class cnn_rd(Module):
    '''
    Class to create the model for the rdn data
    '''
    pass