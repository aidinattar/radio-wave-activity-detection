"""
train.py

This file contains the train function, which is used to train the model.

The model is trained using the Adam optimizer and the Cross Entropy loss.

Usage:
    train.py <model> <data> <input> <case> [--channel=<channel>] (--load|--no-load) [--augment] [--n_samples=<n_samples>] [--aggregate_labels] [--dropout=<dropout>] [--epochs=<epochs>] [--batch_size=<batch_size>] [--optimizer=<optimizer>] [--lr=<lr>] [--weight_decay=<weight_decay>] [--momentum=<momentum>] [--nesterov|--no-nesterov] [--loss=<loss>] [--patience=<patience>] [--min_delta=<min_delta>] [--factor=<factor>] [--verbose=<verbose>] [--seed=<seed>]
    train.py -h | --help

Options:
    -h --help                       Show this screen.
    <model>                         Model to use.
    <data>                          Name of the data file.
    <input>                         Type of data to use.
    <case>                          Case to use.
    --channel=<channel>             Channel to use [default: 1].
    --augment                       Augment the data [default: False].
    --load                          Load the model [default: False].
    --n_samples=<n_samples>         Number of samples to take [default: 5].
    --aggregate_labels              Aggregate the labels [default: False].
    --dropout=<dropout>             Dropout [default: 0.5].
    --epochs=<epochs>               Number of epochs [default: 100].
    --batch_size=<batch_size>       Batch size [default: 32].
    --optimizer=<optimizer>         Optimizer [default: Adam].
    --lr=<lr>                       Learning rate [default: 0.001].
    --weight_decay=<weight_decay>   Weight decay [default: 0.0001].
    --momentum=<momentum>           Momentum [default: 0.9].
    --nesterov                      Nesterov [default: False].
    --loss=<loss>                   Loss [default: CrossEntropyLoss].
    --patience=<patience>           Patience [default: 10].
    --min_delta=<min_delta>         Minimum delta [default: 0.0001].
    --factor=<factor>               Factor [default: 0.1].
    --verbose=<verbose>             Verbose [default: 1].
    --seed=<seed>                   Seed [default: 42].

Example:
    python train.py CNN-MD data_processed.npz mDoppler 2 --load --augment --n_samples=5 --dropout=.2 --epochs=1500 --batch_size=32 --optimizer=Adam --lr=0.001 --weight_decay=0.0001 --momentum=0.9 --nesterov=True --loss=Adam --patience=10 --min_delta=0.0001 --factor=0.1 --verbose=1 --seed=42

Current configuration:
    model: CNN-MD
    data: data_processed.npz
    input: mDoppler
    case: 2
    load: False
    augment: True
    n_samples: 15
    dropout: 0.2
    epochs: 100
    batch_size: 256
    weight_decay: 0.0    
    nesterov: False
    
    python train.py CNN-MD data_processed.npz mDoppler 2  --no-load --augment --n_sample=15 --dropout=.2 --epochs=100 --batch_size=256 --weight_decay=0. --no-nesterov
"""

# TODO:
# Add parameters to the docstring
# Add the parameters not used to the class
# Check the correctness of the code

import torch
import numpy as np
from docopt import docopt
from models.classifier import model
from preprocessing.dataset import Dataset2Channels, Dataset1Channel
from utils.constants import MAPPING_LABELS_DICT
from datetime import datetime
from torchvision import transforms

now = datetime.now().strftime("%Y%m%d")

def main(model_name:str,
         train_data,
         test_data,
         case:int,
         load:bool,
         in_channels:int,
         num_classes:int,
         augment:bool,
         n_samples:int,
         dropout:float,
         epochs:int,
         batch_size:int,
         optimizer:str,
         lr:float,
         weight_decay:float,
         momentum:float,
         nesterov:bool,
         loss:str,
         patience:int,
         min_delta:float,
         factor:float,
         verbose:int,
         device:torch.device,
         seed:int):
    """
    Train the model, save the best model and save the training history

    Parameters
    ----------
    model_name : str
        Name of the model
    data : DataProcess
        DataProcess object containing the data
    case : int
        Case to use
    load : bool
        Load the model
    in_channels : int
        Number of input channels
    num_classes : int
        Number of output channels
    augment : bool
        Augment the data
    n_samples : int
        Number of samples to take in the augmented data
    epochs : int
        Number of epochs
    batch_size : int
        Batch size
    optimizer : str
        Optimizer
    lr : float
        Learning rate
    weight_decay : float
        Weight decay
    momentum : float
        Momentum
    nesterov : bool
        Nesterov
    loss : str
        Loss function
    patience : int
        Patience
    min_delta : float
        Minimum delta
    factor : float
        Factor
    verbose : int
        Verbose
    device : torch.device
        Device to use
    """
    # Create the model object
    classifier = model(
        train_data=train_data,
        test_data=test_data,
        case=case,
        model_type=model_name,
    )
    classifier.create_model(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=dropout
    )

    # Load the pre-trained model
    if load:
        print(f'Loading model {model_name}__case_{case}_checkpoint.pt')
        classifier.load_model(name=f'{model_name}__case_{case}_checkpoint', path='checkpoints')
        
    # Create the DataLoaders
    classifier.create_DataLoaders(batch_size=batch_size)
    
    # Print the model summary
    classifier.summary(
        save=True,
        name=f'{model_name}_case{case}_summary.txt'
    )

    # Create the optimizer, loss function
    classifier.create_optimizer(optimizer=optimizer, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
    classifier.create_loss(
        loss=loss,
        use_weight=True,
    )
    
    #del data, classifier.data, classifier.train_data, classifier.test_data

    # Train the model
    print('Training the model')
    classifier.train_model(epochs=epochs, checkpoint=True, checkpoint_path=f'{model_name}__case_{case}_checkpoint.pt')

    # Plot the training history
    classifier.plot_history(save=True, show=False, save_csv=True)

    # Evaluate the model
    print('Evaluating the model')
    classifier.evaluate_model(save=True)

    # Save the model trained
    print('Saving the model')
    classifier.save_trained_model(name=f'{model_name}_{now}_case{case}')


if __name__ == '__main__':
    args = docopt(__doc__)

    # set seed
    seed = int(args['--seed'])
    torch.manual_seed(seed)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    load = bool(args['--load'])
    augment = bool(args['--augment'])
    n_samples = int(args['--n_samples'])
    aggregate = bool(args['--aggregate_labels'])

    # set hyperparameters
    case = int(args['<case>'])
    dropout = float(args['--dropout'])
    epochs = int(args['--epochs'])
    batch_size = int(args['--batch_size'])
    optimizer = args['--optimizer']
    lr = float(args['--lr'])
    weight_decay = float(args['--weight_decay'])
    momentum = float(args['--momentum'])
    nesterov = bool(args['--nesterov'])
    loss = args['--loss']
    patience = int(args['--patience'])
    min_delta = float(args['--min_delta'])
    factor = float(args['--factor'])
    verbose = int(args['--verbose'])

    TYPE = args['<input>']
    
    
    labels_transform = np.vectorize(
        lambda label: MAPPING_LABELS_DICT[label]
    ) if aggregate else None

    num_classes = 10 if aggregate else 14
    
    
    if case == 0:

        features_transform = transforms.Compose([
            lambda x: x[:, 20:-20],
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))
        ])

        train_data = Dataset1Channel(
            TYPE=TYPE,
            dirname='DATA_preprocessed',
            filename=args['<train_data>'],
            features_transform=features_transform,
            labels_transform=labels_transform,
            channel=1,
        )
        
        test_data = Dataset1Channel(
            TYPE=TYPE,
            dirname='DATA_preprocessed',
            filename=args['<test_data>'],
            features_transform=features_transform,
            labels_transform=labels_transform,
            channel=1,
        )
        in_channels = 1
        
    elif case == 1:
        raise NotImplementedError('Case 1 not implemented yet')

    elif case == 2:
        raise NotImplementedError('Case 2 not implemented yet')
    
    elif case == 3:

        features_transform = transforms.Compose([
            lambda x: x[:, :, 20:-20],
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))
        ])
        
        train_data = Dataset2Channels(
            TYPE=TYPE,
            dirname='DATA_preprocessed',
            filename=args['<train_data>'],
            features_transform=features_transform,
            labels_transform=labels_transform,
            combine_channels=True
        )
        
        test_data = Dataset2Channels(
            TYPE=TYPE,
            dirname='DATA_preprocessed',
            filename=args['<test_data>'],
            features_transform=features_transform,
            labels_transform=labels_transform,
            combine_channels=True
        )
        
        in_channels = 2

    else:
        raise ValueError(f'Case {case} not recognized')
    
    # load model
    model_name = args['<model>']

    main(
        model_name=model_name,
        train_data=train_data,
        test_data=test_data,
        case=case,
        load=load,
        in_channels=in_channels,
        num_classes=num_classes,
        augment=augment,
        n_samples=n_samples,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        loss=loss,
        patience=patience,
        min_delta=min_delta,
        factor=factor,
        verbose=verbose,
        device=device,
        seed=seed
    )