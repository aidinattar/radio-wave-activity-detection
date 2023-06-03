"""
train_gan.py

Trains a GAN.

Usage:
    train_gan.py (--filename <filename>) (--dirname <dirname>) [--channel=<channel>] [--batch_size=<batch_size>] [--epochs=<epochs>] [--lr=<lr>]
    
Options:
    -h --help                       Show this screen.
    -f --filename <filename>        Name of the file to save the model.
    -d --dirname <dirname>          Name of the directory to save the model.
    -c --channel=<channel>          Channel to use [default: 1].
    -b --batch_size=<batch_size>    Batch size [default: 32].
    -l --lr=<lr>                    Learning rate [default: 0.001].
    -e --epochs=<epochs>            Number of epochs [default: 100].

Example:
    python train_gan.py --filename gan.h5 --dirname gan
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
from models.GAN import generator_loss, discriminator_loss


def train_step(images: torch.Tensor,
               generator: torch.nn.Module,
               discriminator: torch.nn.Module,
               BATCH_SIZE: int,
               noise_dim: int,
               device: str,
               dis_opt: torch.optim.Optimizer,
               gen_opt: torch.optim.Optimizer):
    """
    Performs a training step for a GAN
    
    Parameters
    ----------
    images : torch.Tensor
        Batch of images
    generator : torch.nn.Module
        Generator model
    discriminator : torch.nn.Module
        Discriminator model
    BATCH_SIZE : int
        Batch size
    noise_dim : int
        Dimension of the noise vector
    device : str or torch.device
        Device to perform the computation
    dis_opt : torch.optim.Optimizer
        Optimizer for the discriminator
    gen_opt : torch.optim.Optimizer
        Optimizer for the generator
    
    Returns
    -------
    gen_loss : torch.Tensor
        Generator loss
    disc_loss : torch.Tensor
        Discriminator loss
    """

    # generate noise vector    
    noise = torch.randn([BATCH_SIZE, noise_dim], device=device)

    # generate images
    generated_images = generator(noise)

    # forward pass through discriminator
    # for real and fake images
    real_output = discriminator(images)
    fake_output = discriminator(generated_images.detach())

    # compute losses for discriminator
    disc_loss = discriminator_loss(real_output, fake_output, device=device)
    dis_opt.zero_grad()
    disc_loss.backward()
    dis_opt.step()

    # forward pass through discriminator
    fake_output = discriminator(generated_images)
    # compute losses for generator
    gen_loss = generator_loss(fake_output, device=device)
    gen_opt.zero_grad()
    gen_loss.backward()
    gen_opt.step()

    return gen_loss, disc_loss


def generate_and_save_images(model: torch.nn.Module,
                             epoch: int,
                             test_input: torch.Tensor,
                             show: bool=False):
    """
    Generates and saves images from a GAN
    
    Parameters
    ----------
    model : torch.nn.Module
        Generator model
    epoch : int
        Epoch number
    test_input : torch.Tensor
        Noise vector
    show : bool, optional
        Whether to show the images or not
    """
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    model.eval()
    with torch.no_grad():
        predictions = model(test_input).detach().cpu()* 250

    # Plot the generated images
    grid = make_grid(predictions, 4).numpy().squeeze().transpose(1, 2, 0)

    plt.imshow(grid.astype(np.uint8) , cmap='binary')
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            'figures',
            'image_at_epoch_{:04d}.png'.format(epoch)
        )
    )
    
    if show:
        plt.show()

    model.train()
    
    
def train(dataloader: DataLoader,
          epochs: int,
          generator: torch.nn.Module,
          discriminator: torch.nn.Module,
          BATCH_SIZE: int,
          noise_dim: int,
          device: str,
          dis_opt: torch.optim.Optimizer,
          gen_opt: torch.optim.Optimizer,
          seed: torch.Tensor,
          show: bool=False):
    """
    Trains a GAN
    
    Parameters
    ----------
    dataloader : DataLoader
        DataLoader for the dataset
    epochs : int
        Number of epochs
    generator : torch.nn.Module
        Generator model
    discriminator : torch.nn.Module
        Discriminator model
    BATCH_SIZE : int
        Batch size
    noise_dim : int
        Dimension of the noise vector
    device : str or torch.device
        Device to perform the computation
    dis_opt : torch.optim.Optimizer
        Optimizer for the discriminator
    gen_opt : torch.optim.Optimizer
        Optimizer for the generator
    seed : torch.Tensor
        Noise vector
    show : bool, optional
        Whether to show the images or not
        
    Returns
    -------
    gloss : list
        Generator losses
    dloss : list
        Discriminator losses
    """
    gloss = []
    dloss = []
    
    for epoch in tqdm(range(epochs)):
        gen_losses = []
        disc_losses = []
        # Train loop
        for image_batch, _ in dataloader:
            image_batch = image_batch.to(device)
            gen_loss, disc_loss = train_step(image_batch, generator, discriminator, BATCH_SIZE, noise_dim, device, dis_opt, gen_opt)
            gen_losses.append(gen_loss.detach().cpu())
            disc_losses.append(disc_loss.detach().cpu())

        # collect losses
        gloss.append(np.mean(gen_losses))
        dloss.append(np.mean(disc_losses))
        
        # save checkpoint if loss is lower than the previous one
        if gloss[-1] == min(gloss):
            torch.save(generator.state_dict(), os.path.join('checkpoints', 'generator.pt'))
            torch.save(discriminator.state_dict(), os.path.join('checkpoints', 'discriminator.pt'))
        
        # produce images
        display.clear_output(wait=True)
        generate_and_save_images(
            model=generator,
            epoch=epoch + 1,
            test_input=seed,
            show=show
        )

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(
        model=generator,
        epoch=epochs,
        test_input=seed,
        show=show
    )

    return gloss, dloss


if __name__=='__main__':
    
    args = docopt(__doc__)
    
    # labels mapping
    labels_transform = np.vectorize(
        lambda label: MAPPING_LABELS_DICT[label]
    )

    # transforms:
    # 1. cut the first and last 9 rows
    # 2. convert to tensor
    # 3. normalize
    features_transform = transforms.Compose([
        lambda x: x[:,9:-9].T,
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))
    ])
    
    # dataset
    data = Dataset1Channel(
        TYPE='mDoppler',
        dirname=args['--dirname'],
        filename=args['--filename'],
        features_transform=features_transform,
        labels_transform=labels_transform,
        channel=int(args['--channel'])
    )
    
    data_loader = DataLoader(
        dataset=data,
        batch_size=int(args['--batch-size']),
        shuffle=True
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # create the generator
    generator = Generator(
        input_size=100
    )
    
    # create the discriminator
    discriminator = Discriminator()
    
    # optimizers
    gen_opt = torch.optim.Adam(
        generator.parameters(),
        lr=float(args['--lr'])
    )
    
    dis_opt = torch.optim.Adam(
        discriminator.parameters(),
        lr=float(args['--lr'])
    )
    
    EPOCHS = int(args['--epochs'])
    noise_dim = 100
    num_examples_to_generate = 16
    # seed for the generator
    seed = torch.randn(
         [num_examples_to_generate, noise_dim],
         device=device
    )
    
    # train the GAN
    generator.to(device)
    discriminator.to(device)
    gloss, dloss = train(
        dataloader=data_loader,
        epochs=EPOCHS,
        generator=generator,
        discriminator=discriminator,
        BATCH_SIZE=int(args['--batch-size']),
        noise_dim=noise_dim,
        device=device,
        dis_opt=dis_opt,
        gen_opt=gen_opt,
        seed=seed,
        show=False
    )
    
    # save the models
    torch.save(
        generator.state_dict(),
        os.path.join(
            'trained_models',
            'generator.pt'
        )
    )
    
    torch.save(
        discriminator.state_dict(),
        os.path.join(
            'trained_models',
            'discriminator.pt'
        )
    )
    
    # save the losses
    np.save(
        os.path.join(
            'logs',
            'gloss.npy'
        ),
        np.array(gloss)
    )
    
    np.save(
        os.path.join(
            'logs',
            'dloss.npy'
        ),
        np.array(dloss)
    )
    
    # plot the losses
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gloss, label="Generator Loss")
    ax.plot(dloss, label="Discriminator Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Generator and Discriminator Loss Over Time")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', which='both', labelsize=10)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            'figures',
            'losses_gan.png'
        )
    )
            
    plt.show()