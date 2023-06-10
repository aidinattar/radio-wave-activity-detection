import torch
from torch.nn import Module, Linear,\
    Conv2d, MaxPool2d, BatchNorm2d,\
    ReLU, AdaptiveAvgPool2d, Softmax, Dropout


class MainPath(Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.main_path = torch.nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            BatchNorm2d(out_channels),
            ReLU(),
            Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.main_path(x)


class IdentityBlock(Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.identity_block = torch.nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm2d(out_channels),
            ReLU(),
            Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return x + self.identity_block(x)


class ResNet18(Module):

    def __init__(self, in_channels, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU()
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(512, num_classes)
        self.softmax = Softmax(dim=1)
        self.dropout = Dropout(dropout)

        self._init_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [MainPath(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(IdentityBlock(out_channels, out_channels))
        return torch.nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)  # Apply dropout before the fully connected layer
        x = self.fc(x)
        x = self.softmax(x)

        return x
