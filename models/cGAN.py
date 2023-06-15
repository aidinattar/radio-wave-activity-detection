"""
Conditional Generative Adversarial Network (cGAN) model
"""

import torch
from torch.nn import Module, Linear,\
    Conv2d, MaxPool2d, Dropout, Dropout2d,\
    Flatten, Sequential, ELU, Softmax,\
    BatchNorm1d, BatchNorm2d, LeakyReLU,\
    ConvTranspose2d, Sigmoid, Embedding, Tanh
import numpy as np


map_shape = (1, 80, 40)
n_classes = 10

class Generator(Module):
    def __init__(self, n_classes, latent_dim):
        super().__init__()

        self.label_emb = Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [Linear(in_feat, out_feat)]
            if normalize:
                layers.append(BatchNorm1d(out_feat, 0.8))
            layers.append(LeakyReLU(0.2, inplace=True))
            return layers

        self.model = Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            Linear(1024, int(np.prod(map_shape))),
            Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *map_shape)
        return img



class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = Embedding(n_classes, n_classes)

        self.model = Sequential(
            Linear(n_classes + int(np.prod(map_shape)), 512),
            LeakyReLU(0.2, inplace=True),
            Linear(512, 512),
            Dropout(0.4),
            LeakyReLU(0.2, inplace=True),
            Linear(512, 512),
            Dropout(0.4),
            LeakyReLU(0.2, inplace=True),
            Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity