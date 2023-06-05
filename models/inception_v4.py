import torch
from torch.nn import Module, Linear,\
                     Conv2d, MaxPool2d,\
                     Dropout, Dropout2d,\
                     Flatten, Sequential,\
                     ELU, Softmax,\
                     BatchNorm2d, ReLU,\
                     AdaptiveAvgPool2d, AvgPool2d
import torch.nn.functional as F

               
# convolutional and batch normalization helper function      
class Conv2d_bn(Module):

    def __init__(self, in_filters, out_filters, kernel_size, strides, padding):
        super().__init__()
        if isinstance(kernel_size, tuple):
            padding_val = (k // 2 for k in kernel_size) if padding == "same" else (0,0)
        else:
            padding_val = kernel_size // 2 if padding == "same" else 0
        self.conv = Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=strides, padding=padding_val)
        self.bn = BatchNorm2d(out_filters)
        self.relu = ReLU()
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
        return self.relu(self.bn(self.conv(x)))
    

# stem block
class StemBlock(Module):

    def __init__(
        self,
        in_filters:int=2
    ):
        super().__init__()
        self.first_block = Sequential(
            Conv2d_bn(in_filters=in_filters, out_filters=32, kernel_size=3, strides=2, padding="valid"),
            Conv2d_bn(in_filters=32, out_filters=32, kernel_size=3, strides=1, padding="valid"),
            Conv2d_bn(in_filters=32, out_filters=64, kernel_size=3, strides=1, padding="same"),
        )
        self.first_left = MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.first_right = Conv2d_bn(in_filters=64, out_filters=96, kernel_size=3, strides=2, padding="valid")
        self.second_left =  Sequential(
            Conv2d_bn(in_filters=160, out_filters=64, kernel_size=1, strides=1, padding="same"),
            Conv2d_bn(in_filters=64, out_filters=96, kernel_size=3, strides=1, padding="valid"),
        )
        self.second_right =  Sequential(
            Conv2d_bn(in_filters=160, out_filters=64, kernel_size=1, strides=1, padding="same"),
            Conv2d_bn(in_filters=64, out_filters=64, kernel_size=(7, 1), strides=1, padding="same"),
            Conv2d_bn(in_filters=64, out_filters=64, kernel_size=(1, 7), strides=1, padding="same"),
            Conv2d_bn(in_filters=64, out_filters=96, kernel_size=3, strides=1, padding="valid"),
        )
        self.third_left = Conv2d_bn(in_filters=192, out_filters=192, kernel_size=3, strides=2, padding="valid")
        self.third_right = MaxPool2d(kernel_size=3, stride=2, padding=0)
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
        x = self.first_block(x)
        x_l = self.first_left(x)
        x_r = self.first_right(x)
        x = torch.cat([x_l, x_r], axis=1)
        x_l = self.second_left(x)
        x_r = self.second_right(x)
        x = torch.cat([x_l, x_r], axis=1)
        x_l = self.third_left(x)
        x_r = self.third_right(x)
        x = torch.cat([x_l, x_r], axis=1)
        return x
    
    
# inception A block
class A_block(Module):

    def __init__(self, in_filters):
        super().__init__()
        self.avg_block = Sequential(
            AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv2d_bn(in_filters=in_filters, out_filters=96, kernel_size=1, strides=1, padding="same"),
        )
        self.one_by_one_block = Conv2d_bn(in_filters=in_filters, out_filters=96, kernel_size=1, strides=1, padding="same")
        self.three_by_three_block =  Sequential(
            Conv2d_bn(in_filters=in_filters, out_filters=64, kernel_size=1, strides=1, padding="same"),
            Conv2d_bn(in_filters=64, out_filters=96, kernel_size=3, strides=1, padding="same"),
        )
        self.five_by_five =  Sequential(
            Conv2d_bn(in_filters=in_filters, out_filters=64, kernel_size=1, strides=1, padding="same"),
            Conv2d_bn(in_filters=64, out_filters=96, kernel_size=3, strides=1, padding="same"),
            Conv2d_bn(in_filters=96, out_filters=96, kernel_size=3, strides=1, padding="same"),
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
        x_1 = self.avg_block(x)
        x_2 = self.one_by_one_block(x)
        x_3 = self.three_by_three_block(x)
        x_4 = self.five_by_five(x)
        x = torch.cat([x_1, x_2, x_3, x_4], axis=1)
        return x
    
    
# inception B block
class B_block(Module):

    def __init__(self, in_filters):
        super().__init__()
        self.avg_block = Sequential(
            AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv2d_bn(in_filters=in_filters, out_filters=128, kernel_size=1, strides=1, padding="same"),
        )
        self.one_by_one_block = Conv2d_bn(in_filters=in_filters, out_filters=384, kernel_size=1, strides=1, padding="same")

        self.seven_by_seven_block =  Sequential(
            Conv2d_bn(in_filters=in_filters, out_filters=192, kernel_size=1, strides=1, padding="same"),
            Conv2d_bn(in_filters=192, out_filters=224, kernel_size=(1, 7), strides=1, padding="same"),
            Conv2d_bn(in_filters=224, out_filters=256, kernel_size=(7, 1), strides=1, padding="same"),
        )

        self.thirteen_by_thirteen_block =  Sequential(
            Conv2d_bn(in_filters=in_filters, out_filters=192, kernel_size=1, strides=1, padding="same"),
            Conv2d_bn(in_filters=192, out_filters=192, kernel_size=(1, 7), strides=1, padding="same"),
            Conv2d_bn(in_filters=192, out_filters=224, kernel_size=(7, 1), strides=1, padding="same"),
            Conv2d_bn(in_filters=224, out_filters=224, kernel_size=(1, 7), strides=1, padding="same"),
            Conv2d_bn(in_filters=224, out_filters=256, kernel_size=(7, 1), strides=1, padding="same"),
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
        x_1 = self.avg_block(x)
        x_2 = self.one_by_one_block(x)
        x_3 = self.seven_by_seven_block(x)
        x_4 = self.thirteen_by_thirteen_block(x)
        x = torch.cat([x_1, x_2, x_3, x_4], axis=1)
        return x
    
    
# inception C block
class C_block(Module):

    def __init__(self, in_filters):
        super().__init__()
        self.avg_block = Sequential(
            AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv2d_bn(in_filters=in_filters, out_filters=256, kernel_size=1, strides=1, padding="same"),
        )
        self.one_by_one_block = Conv2d_bn(in_filters=in_filters, out_filters=256, kernel_size=1, strides=1, padding="same")

        self.branch_a =  Conv2d_bn(in_filters=in_filters, out_filters=384, kernel_size=1, strides=1, padding="same")
        self.branch_a_left = Conv2d_bn(in_filters=384, out_filters=256, kernel_size=(1, 3), strides=1, padding="same")
        self.branch_a_right = Conv2d_bn(in_filters=384, out_filters=256, kernel_size=(3, 1), strides=1, padding="same")

        self.branch_b =  Sequential(
            Conv2d_bn(in_filters=in_filters, out_filters=384, kernel_size=1, strides=1, padding="same"),
            Conv2d_bn(in_filters=384, out_filters=448, kernel_size=(1, 3), strides=1, padding="same"),
            Conv2d_bn(in_filters=448, out_filters=512, kernel_size=(3, 1), strides=1, padding="same"),
        )

        self.branch_b_left = Conv2d_bn(in_filters=512, out_filters=256, kernel_size=(1, 3), strides=1, padding="same")
        self.branch_b_right = Conv2d_bn(in_filters=512, out_filters=256, kernel_size=(3, 1), strides=1, padding="same")
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
        x_1 = self.avg_block(x)
        x_2 = self.one_by_one_block(x)
        x_a = self.branch_a(x)
        x_3 = self.branch_a_left(x_a)
        x_4 = self.branch_a_right(x_a)
        x_b = self.branch_b(x)
        x_5 = self.branch_b_left(x_b)
        x_6 = self.branch_b_right(x_b)
        x = torch.cat([x_1, x_2, x_3, x_4, x_5, x_6], axis=1)
        return x
    
    
# reduction A block
class Reduction_A(Module):

    def __init__(self, in_filters):
        super().__init__()
        self.max_pool = MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.central_block = Conv2d_bn(in_filters=in_filters, out_filters=384, kernel_size=3, strides=2, padding="valid")
        self.right_block =  Sequential(
            Conv2d_bn(in_filters=in_filters, out_filters=192, kernel_size=1, strides=1, padding="same"),
            Conv2d_bn(in_filters=192, out_filters=224, kernel_size=3, strides=1, padding="same"),
            Conv2d_bn(in_filters=224, out_filters=256, kernel_size=3, strides=2, padding="valid"),
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
        x_1 = self.max_pool(x)
        x_2 = self.central_block(x)
        x_3 = self.right_block(x)
        x = torch.cat([x_1, x_2, x_3], axis=1)
        return x
    
    
# reduction B block
class Reduction_B(Module):

    def __init__(self, in_filters):
        super().__init__()
        self.max_pool = MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.central_block = Sequential(
            Conv2d_bn(in_filters=in_filters, out_filters=192, kernel_size=1, strides=1, padding="same"),
            Conv2d_bn(in_filters=192, out_filters=192, kernel_size=3, strides=2, padding="valid"),
        )
        self.right_block =  Sequential(
            Conv2d_bn(in_filters=in_filters, out_filters=256, kernel_size=1, strides=1, padding="same"),
            Conv2d_bn(in_filters=256, out_filters=256, kernel_size=(1, 7), strides=1, padding="same"),
            Conv2d_bn(in_filters=256, out_filters=320, kernel_size=(7, 1), strides=1, padding="same"),
            Conv2d_bn(in_filters=320, out_filters=320, kernel_size=3, strides=2, padding="valid"),
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
        x_1 = self.max_pool(x)
        x_2 = self.central_block(x)
        x_3 = self.right_block(x)
        x = torch.cat([x_1, x_2, x_3], axis=1)
        return x
    
    
# inception v4
class InceptionV4(Module):

    def __init__(self,
                 in_channels:int,
                 num_classes:int):
        super().__init__()
        self.stem = StemBlock(in_channels)
        self.inception_a = Sequential(
            A_block(384),
            A_block(384),
            A_block(384),
            A_block(384)
        )
        self.reduction_a = Reduction_A(384)
        self.inception_b = Sequential(
            B_block(1024),
            B_block(1024),
            B_block(1024),
            B_block(1024),
            B_block(1024),
            B_block(1024),
            B_block(1024)
        )
        self.reduction_b = Reduction_B(1024)
        self.inception_c = Sequential(
            C_block(1536),
            C_block(1536),
            C_block(1536)
        )
        self.drop = Dropout(0.2)
        self.out = Linear(1536, num_classes)
        self.apply(self._init_weights)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.reduction_a(x)
        x = self.inception_b(x)
        x = self.reduction_b(x)
        x = self.inception_c(x)
        x = x.reshape(x.shape[0], -1, 1536).mean(axis=1)
        x = self.drop(x)
        y = self.out(x)
        y = F.softmax(y, dim=1)
        return y

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()