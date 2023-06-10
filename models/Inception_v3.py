#%%
import torch
from torch import nn
from torch.nn import Module, Conv2d, Sequential, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Dropout, Linear, Softmax

#%%
class InceptionV3(Module):
    def __init__(self, in_channels=2, num_classes=10, dropout=.2):
        super().__init__()
        self.stem = Sequential(
            BasicConv2d(in_channels, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3),
            BasicConv2d(32, 64, kernel_size=3, padding=1),
            MaxPool2d(kernel_size=3, stride=2)
        )

        self.inception3a = InceptionModule(64, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.inception3c = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception3d = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception3e = InceptionModule(512, 128, 128, 256, 24, 64, 64)

        self.aux1 = InceptionAux(512, num_classes)

        self.inception4a = InceptionModule(512, 256, 160, 320, 32, 128, 128)
        self.inception4b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        self.aux2 = InceptionAux(1024, num_classes)

        self.inception5a = InceptionModule(1024, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(1024, 384, 192, 384, 48, 128, 128)

        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(dropout)
        self.fc = Linear(1024, num_classes)
        self.softmax = Softmax(dim=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.inception3c(x)
        x = self.inception3d(x)
        x = self.inception3e(x)

        aux1 = self.aux1(x)

        x = self.inception4a(x)
        x = self.inception4b(x)

        aux2 = self.aux2(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x
#%%
# Test the model
model = InceptionV3(in_channels=2, num_classes=10)
input_tensor = torch.randn(2, 2, 80, 40)
output = model(input_tensor)
print(output.shape)
# %%
