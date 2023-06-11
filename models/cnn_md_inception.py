import torch
from torch.nn import Module, Linear,\
    Conv2d, MaxPool2d, Dropout, Dropout2d,\
    Flatten, Sequential, ELU, Softmax, ReLU

class cnn_md_inception(Module):
    """
    Class to create the model for the mDoppler data
    """

    def __init__(self,
                 in_channels: int=1,
                 out_channels: int=10,
                 filters: tuple=(10, 16, 16, 64),
                 pool_size: tuple=(2, 2),
                 pool_stride: int=2,
                 pool_padding: int=1,
                 dilation: int=1,
                 bias: bool=True,
                 dropout: float=0.5,
                 ):
        """
        Constructor
        """
        super().__init__()
        f1, f2, f3, f4 = filters

        # Convolutional layers with Inception modules
        self.cnn = Sequential(

            InceptionModule(
                in_channels=in_channels,
                out_channels=f1,
                kernel_sizes=[1, 3, 5],
                bias=bias
            ),
            MaxPool2d(kernel_size=pool_size,
                      stride=pool_stride,
                      padding=pool_padding,
                      dilation=dilation),

            Dropout2d(p=dropout),

            InceptionModule(
                in_channels=f1*3+2,
                out_channels=f2,
                kernel_sizes=[1, 3, 5],
                bias=bias
            ),
            MaxPool2d(
                kernel_size=pool_size,
                stride=pool_stride,
                padding=pool_padding,
                dilation=dilation
            ),

            Dropout2d(p=dropout),

            InceptionModule(
                in_channels=f2*3+32,
                out_channels=f3,
                kernel_sizes=[1, 3, 5],
                bias=bias
            ),
            MaxPool2d(
                kernel_size=pool_size,
                stride=pool_stride,
                padding=pool_padding,
                dilation=dilation
            ),

            Dropout2d(p=dropout),

            InceptionModule(
                in_channels=f3*3+80,
                out_channels=f4,
                kernel_sizes=[1, 3, 5],
                bias=bias
            ),
            MaxPool2d(kernel_size=pool_size,
                      stride=pool_stride,
                      padding=pool_padding,
                      dilation=dilation),

        )

        # Flatten the output of the convolutional layers
        self.flatten = Flatten()

        # Fully connected layers
        self.fc = Sequential(
            # num_filters * height * width
            # height = (input_height - kernel_size + 2 * padding) / stride + 1
            # width = (input_width - kernel_size + 2 * padding) / stride + 1
            Linear(in_features=(f4*3+128)*4*6, out_features=128),  # 6 and 4 are the height and width of the input
            ReLU(),
            Dropout(p=dropout),
            Linear(in_features=1280, out_features=128),
            ReLU(),
            Dropout(p=dropout),
            Linear(in_features=128, out_features=out_channels),
            Softmax(dim=1)
        )

        self.apply(self._init_weights)

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

    def forward(self, x):
        """
        Forward pass
        """
        # Convolutional layers
        x = self.cnn(x)
        # Flatten the output of the convolutional layers
        x = self.flatten(x)
        # Fully connected layers
        x = self.fc(x)
        return x

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


class InceptionModule(Module):
    """
    Inception module implementation
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: list, bias: bool = True):
        """
        Constructor
        """
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=kernel_sizes[0], bias=bias)
        self.conv3 = Conv2d(in_channels, out_channels, kernel_size=kernel_sizes[1], padding=1, bias=bias)
        self.conv5 = Conv2d(in_channels, out_channels, kernel_size=kernel_sizes[2], padding=2, bias=bias)
        self.maxpool = MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.activation = ELU()

    def forward(self, x):
        """
        Forward pass
        """
        out1 = self.activation(self.conv1(x))
        out3 = self.activation(self.conv3(x))
        out5 = self.activation(self.conv5(x))
        out_pool = self.activation(self.maxpool(x))
        out = torch.cat((out1, out3, out5, out_pool), dim=1)
        return out