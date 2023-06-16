"""
train_gan.py

Trains a GAN.

Usage:
    train_gan.py [--epochs=<epochs>] [--batch_size=<batch_size>] [--lr=<lr>] [--b1=<b1>] [--b2=<b2>] [--img_height=<img_height>] [--img_width=<img_width>] [--channels=<channels>] [--latent_dim=<latent_dim>] [--sample_interval=<sample_interval>]
        
Options:
    -h --help                       Show this screen.
    --epochs=<epochs>               Number of epochs [default: 200].
    --batch_size=<batch_size>       Batch size [default: 64].
    --lr=<lr>                       Learning rate [default: 0.0002].
    --b1=<b1>                       Adam: decay of first order momentum of gradient [default: 0.5].
    --b2=<b2>                       Adam: decay of first order momentum of gradient [default: 0.999].
    --img_height=<img_height>       Height of the images [default: 80].
    --img_width=<img_width>         Width of the images [default: 40].
    --channels=<channels>           Number of image channels [default: 1].
    --latent_dim=<latent_dim>       Dimensionality of the latent space [default: 100].
    --sample_interval=<sample_interval>   Interval between image samples [default: 400].
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from docopt import docopt
from IPython import display
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from utils.constants import MAPPING_LABELS_DICT
from preprocessing.dataset import Dataset1Channel
from models.GAN import Generator, Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image

os.makedirs("images", exist_ok=True)

opt = docopt(__doc__)

img_shape = (
    int(opt['--channels']),
    int(opt['--img_height']),
    int(opt['--img_width']),
)

cuda = True if torch.cuda.is_available() else False

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
# labels mapping
labels_transform = np.vectorize(
    lambda label: MAPPING_LABELS_DICT[label]
)

# transforms:
# 1. cut the first and last 9 rows
# 2. convert to tensor
# 3. normalize
features_transform = transforms.Compose([
    #lambda x: x[:,9:-9].T,
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])

# dataset
data = Dataset1Channel(
    TYPE='mDoppler',
    dirname='DATA_preprocessed',
    filename='processed_data_mDoppler.h5',
    features_transform=features_transform,
    labels_transform=labels_transform,
    channel=1
)

dataloader = DataLoader(
    dataset=data,
    batch_size=int(opt['--batch_size']),
    shuffle=True
)


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=float(opt['--lr']), betas=(float(opt['--b1']), float(opt['--b2'])))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=float(opt['--lr']), betas=(float(opt['--b1']), float(opt['--b2'])))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(int(opt['--epochs'])):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], int(opt['--latent_dim'])))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, int(opt['--epochs']), i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % int(opt['--sample_interval']) == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            # save checkpoints
            torch.save(generator.state_dict(), 'checkpoints/generatorGAN.pth')
            torch.save(discriminator.state_dict(), 'checkpoints/discriminatorGAN.pth')
            
# Save models checkpoints
torch.save(generator.state_dict(), 'trained_models/generatorGAN.pth')
torch.save(discriminator.state_dict(), 'trained_models/discriminatorGAN.pth')
