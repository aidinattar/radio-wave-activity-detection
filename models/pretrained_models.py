"""
pretrained_models.py
"""

import torchvision.models as models
from torch import nn

def InceptionV3(
    input_size=(2,40,80),
    num_classes:int=5,
    pretrained:bool=True,
    aux_logits:bool=False
)->nn.Module:
    # Load and modify InceptionV3
    model = models.inception_v3(
        pretrained=pretrained,
        aux_logits=aux_logits
    )
    # Modify input layer to accept n channels
    model.Conv2d_1a_3x3.conv = nn.Conv2d(
        in_channels=input_size[0],
        out_channels=32,
        kernel_size=(3,3),
        stride=(2,2),
        bias=False,
        padding=(0,0)
    )
    
    # Modify output layer to output n classes
    num_features = model.fc.in_features
    
    model.fc = nn.Sequential(
        nn.Linear(num_features, num_classes),
        nn.Softmax(dim=1)
    )

    return model


def InceptionResNetV2(
    input_size=(2,40,80),
    num_classes:int=5,
    pretrained:bool=True,
    aux_logits:bool=False
)->nn.Module:
    
    # Load and modify InceptionResNetV2
    model = models.inceptionresnet_v2(
        pretrained=pretrained,
        aux_logits=aux_logits
    )
    # Modify input layer to accept n channels
    model.conv2d_1a.conv = nn.Conv2d(
        in_channels=input_size[0],
        out_channels=32,
        kernel_size=(3,3),
        stride=(2,2),
        bias=False,
        padding=(0,0)
    )
    # Modify output layer to output n classes
    num_features = model.last_linear.in_features
    model.last_linear = nn.Sequential(
        nn.Linear(num_features, num_classes),
        nn.Softmax(dim=1)
    )

    return model