#%%
import torch
from torch import nn
from torch.nn import Module, Conv2d, Sequential, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Dropout, Linear, Dropout2d, Flatten, Softmax

#%%
class BasicConv2d(Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, **kwargs)
        self.relu = ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class InceptionModule(Module):
    def __init__(self, in_channels, conv1x1, reduce3x3, conv3x3, reduce5x5, conv5x5, reduce_pool):
        super().__init__()

        self.branch1 = BasicConv2d(in_channels, conv1x1, kernel_size=1)

        self.branch2 = Sequential(
            BasicConv2d(in_channels, reduce3x3, kernel_size=1),
            BasicConv2d(reduce3x3, conv3x3, kernel_size=3, padding=1)
        )

        self.branch3 = Sequential(
            BasicConv2d(in_channels, reduce5x5, kernel_size=1),
            BasicConv2d(reduce5x5, conv5x5, kernel_size=3, padding=1),
            BasicConv2d(conv5x5, conv5x5, kernel_size=3, padding=1)
        )

        self.branch4 = Sequential(
            MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, reduce_pool, kernel_size=1)
        )

    def forward(self, x):
        branch1_output = self.branch1(x)
        branch2_output = self.branch2(x)
        branch3_output = self.branch3(x)
        branch4_output = self.branch4(x)

        output = torch.cat([branch1_output, branch2_output, branch3_output, branch4_output], 1)
        return output


class InceptionReduction(Module):
    def __init__(self, in_channels, reduce3x3, conv3x3_reduce, conv3x3, reduce_pool):
        super().__init__()

        self.branch1 = BasicConv2d(in_channels, reduce3x3, kernel_size=1)
        self.branch2 = Sequential(
            BasicConv2d(in_channels, conv3x3_reduce, kernel_size=1),
            BasicConv2d(conv3x3_reduce, conv3x3, kernel_size=3, stride=2)
        )
        self.branch3 = Sequential(
            BasicConv2d(in_channels, reduce_pool, kernel_size=1),
            MaxPool2d(kernel_size=3, stride=2)
        )

    def forward(self, x):
        branch1_output = self.branch1(x)
        branch2_output = self.branch2(x)
        branch3_output = self.branch3(x)

        output = torch.cat([branch1_output, branch2_output, branch3_output], 1)
        return output


class InceptionResNetA(Module):
    def __init__(self, in_channels, scale=1.0):
        super().__init__()

        self.scale = scale
        self.branch1 = BasicConv2d(in_channels, 32, kernel_size=1)

        self.branch2 = Sequential(
            BasicConv2d(in_channels, 32, kernel_size=1),
            BasicConv2d(32, 32, kernel_size=3, padding=1)
        )

        self.branch3 = Sequential(
            BasicConv2d(in_channels, 32, kernel_size=1),
            BasicConv2d(32, 48, kernel_size=3, padding=1),
            BasicConv2d(48, 64, kernel_size=3, padding=1)
        )

        self.conv = Conv2d(128, in_channels, kernel_size=1)
        self.relu = ReLU(inplace=True)

    def forward(self, x):
        branch1_output = self.branch1(x)
        branch2_output = self.branch2(x)
        branch3_output = self.branch3(x)

        output = torch.cat([branch1_output, branch2_output, branch3_output], 1)
        output = self.conv(output)
        output = output * self.scale + x
        output = self.relu(output)
        return output


class InceptionResNetB(Module):
    def __init__(self, in_channels, scale=1.0):
        super().__init__()

        self.scale = scale
        self.branch1 = BasicConv2d(in_channels, 192, kernel_size=1)

        self.branch2 = Sequential(
            BasicConv2d(in_channels, 128, kernel_size=1),
            BasicConv2d(128, 160, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(160, 192, kernel_size=(7, 1), padding=(3, 0))
        )

        self.conv = Conv2d(384, in_channels, kernel_size=1)
        self.relu = ReLU(inplace=True)

    def forward(self, x):
        branch1_output = self.branch1(x)
        branch2_output = self.branch2(x)

        output = torch.cat([branch1_output, branch2_output], 1)
        output = self.conv(output)
        output = output * self.scale + x
        output = self.relu(output)
        return output


class InceptionResNetC(Module):
    def __init__(self, in_channels, scale=1.0):
        super().__init__()

        self.scale = scale
        self.branch1 = BasicConv2d(in_channels, 192, kernel_size=1)

        self.branch2 = Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 224, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(224, 256, kernel_size=(3, 1), padding=(1, 0))
        )

        self.conv = Conv2d(448, in_channels, kernel_size=1)
        self.relu = ReLU(inplace=True)

    def forward(self, x):
        branch1_output = self.branch1(x)
        branch2_output = self.branch2(x)

        output = torch.cat([branch1_output, branch2_output], 1)
        output = self.conv(output)
        output = output * self.scale + x
        output = self.relu(output)
        return output


class InceptionResNetReduction(Module):
    def __init__(self, in_channels, reduce3x3, conv3x3_reduce, conv3x3, reduce_pool):
        super().__init__()

        self.branch1 = BasicConv2d(in_channels, reduce3x3, kernel_size=1)
        self.branch2 = Sequential(
            BasicConv2d(in_channels, conv3x3_reduce, kernel_size=1),
            BasicConv2d(conv3x3_reduce, conv3x3, kernel_size=3, stride=2)
        )
        self.branch3 = Sequential(
            BasicConv2d(in_channels, reduce_pool, kernel_size=1),
            MaxPool2d(kernel_size=3, stride=2)
        )

    def forward(self, x):
        branch1_output = self.branch1(x)
        branch2_output = self.branch2(x)
        branch3_output = self.branch3(x)

        output = torch.cat([branch1_output, branch2_output, branch3_output], 1)
        return output


class InceptionResNetV2(Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 dropout=0.6):
        super().__init__()

        self.stem = Sequential(
            BasicConv2d(in_channels, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3, stride=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            MaxPool2d(kernel_size=3, stride=2)
        )

        self.inceptionA = Sequential(
            InceptionResNetA(64),
            InceptionResNetA(384),
            InceptionResNetA(384),
        )

        self.reductionA = InceptionResNetReduction(384, 192, 192, 256, 384)

        self.inceptionB = Sequential(
            InceptionResNetB(1152),
            InceptionResNetB(1152),
            InceptionResNetB(1152),
            InceptionResNetB(1152),
            InceptionResNetB(1152),
            InceptionResNetB(1152),
            InceptionResNetB(1152),
        )

        self.reductionB = InceptionResNetReduction(1152, 256, 256, 384, 256)

        self.inceptionC = Sequential(
            InceptionResNetC(2048),
            InceptionResNetC(2048),
            InceptionResNetC(2048),
        )

        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(dropout)
        self.fc = Linear(2048, num_classes)
        self.softmax = Softmax(dim=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.inceptionA(x)
        x = self.reductionA(x)
        x = self.inceptionB(x)
        x = self.reductionB(x)
        x = self.inceptionC(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x
#%%
model = InceptionResNetV2(in_channels=2, num_classes=10)
input_tensor = torch.randn(2, 2, 80, 40)
output = model(input_tensor)
print(output.shape)
# %%
