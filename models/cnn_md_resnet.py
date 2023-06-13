import torch
from torch.nn import Module, Linear,\
    Conv2d, MaxPool2d, Dropout, Dropout2d,\
    Flatten, Sequential, ELU, Softmax,\
    ReLU, BatchNorm2d, BatchNorm1d, ModuleList
import torch.nn.functional as F

class cnn_md_resnet_v1(Module):
    """
    Class to create the model for the mDoppler data
    """

    def __init__(self,
                 in_channels: int=1,
                 out_channels: int=10,
                 filters: tuple=(8, 16, 32, 64),
                 kernel_size: tuple=(3, 3),
                 stride: int=1,
                 padding: str='same',
                 pool_size: tuple=(2, 2),
                 pool_stride: int=2,
                 pool_padding: int=1,
                 dilation: int=1,
                 groups: int=1,
                 bias: bool=True,
                 padding_mode: str='zeros',
                 dropout: float=0.5,
                 ):
        """
        Constructor
        """
        super().__init__()
        f1, f2, f3, f4 = filters

        # Convolutional layers
        self.cnn = Sequential(

            Conv2d(in_channels=in_channels,
                   out_channels=f1,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   groups=groups,
                   bias=bias,
                   dilation=dilation,
                   padding_mode=padding_mode),
            ELU(),
            MaxPool2d(kernel_size=pool_size,
                      stride=pool_stride,
                      padding=pool_padding,
                      dilation=dilation),

            Dropout2d(p=dropout),

            Conv2d(in_channels=f1,
                   out_channels=f2,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   groups=groups,
                   bias=bias,
                   dilation=dilation,
                   padding_mode=padding_mode),
            ELU(),
            MaxPool2d(kernel_size=pool_size,
                      stride=pool_stride,
                      padding=pool_padding,
                      dilation=dilation),

            Dropout2d(p=dropout),

            Conv2d(in_channels=f2,
                   out_channels=f3,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   groups=groups,
                   bias=bias,
                   dilation=dilation,
                   padding_mode=padding_mode),
            ELU(),
            MaxPool2d(kernel_size=pool_size,
                      stride=pool_stride,
                      padding=pool_padding,
                      dilation=dilation),

            Dropout2d(p=dropout),

            Conv2d(in_channels=f3,
                   out_channels=f4,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   groups=groups,
                   bias=bias,
                   dilation=dilation,
                   padding_mode=padding_mode),
            ELU(),
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
            Linear(in_features=f4*4*6, out_features=128),  # 6 and 4 are the height and width of the input
            ReLU(),
            Dropout(p=dropout),
            Linear(in_features=128, out_features=out_channels),
            Softmax(dim=1)
        )

        self.shortcut = Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=f4,
                kernel_size=(1, 1),
                stride=stride,
                padding=0,
                bias=False
            ),
            BatchNorm2d(f4)
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
        residual = self.shortcut(x)
        x = self.cnn(x)
        residual = F.interpolate(residual, size=x.size()[2:], mode='bilinear', align_corners=False)
        x += residual  # Add residual connection
        x = ELU()(x)
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



####################################
# CNN-MD-ResNet-V2 model from here #
####################################
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

class cnn_md_resnet_v2(Module):
    """
    Class to create the model for the mDoppler data
    """

    def __init__(self,
                 in_channels: int=1,
                 out_channels: int=10,
                 filters: tuple=(8, 16, 32, 64),
                 neurons: tuple=(128, 64),
                 kernel_size=3,
                 stride: int=1,
                 padding: str='same',
                 pool_size=2,
                 pool_stride: int=2,
                 pool_padding: int=1,
                 dilation: int=1,
                 padding_mode: str='zeros',
                 dropout: float=0.5,
                 ):
        """
        Constructor
        """
        super().__init__()

        self.shortcut_block = ModuleList()
        
        in_ch = in_channels
        for i, num_filters in enumerate(filters):
            self.shortcut_block.append(
                Sequential(
                    Conv2d(
                        in_channels=in_ch,
                        out_channels=num_filters,
                        kernel_size=(1, 1),
                        stride=stride,
                        padding=0,
                        bias=False
                    ),
                    BatchNorm2d(num_filters)
                )
            )
            in_ch = num_filters
    
        self.pool = Sequential(
            MaxPool2d(
                kernel_size=pool_size,
                stride=pool_stride,
                padding=pool_padding
            ),
            Dropout2d(
                p=dropout
            )
        )
    
        self.conv_blocks = ModuleList()
        in_ch = in_channels
        for i, num_filters in enumerate(filters):
            self.conv_blocks.append(
                ConvBlock(
                    in_channels=in_ch,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                )
            )
            in_ch = num_filters

        # Flatten the output of the convolutional layers
        self.flatten = Flatten()

        # Fully connected layers
        self.linear_blocks = ModuleList()
        in_features = filters[-1] * 6 * 4
        for num_neurons in neurons:
            self.linear_blocks.append(
                LinearBlock(
                    in_features=in_features,
                    out_features=num_neurons,
                    dropout=dropout,
                )
            )
            in_features = num_neurons

        
        self.output_layer = Sequential(
            Linear(
                in_features=in_features,
                out_features=out_channels
            ),
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
        for conv, shortcut in zip(self.conv_blocks, self.shortcut_block):
            residual = shortcut(x)
            x = conv(x)
            x += residual  # Add residual connection
            x = self.pool(x)    
        x = ELU()(x)
        # Flatten the output of the convolutional layers
        x = self.flatten(x)
        # Fully connected layers
        for linear in self.linear_blocks:
            x = linear(x)
        x = self.output_layer(x)
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
