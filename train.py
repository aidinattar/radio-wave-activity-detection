"""
train.py

This file contains the train function, which is used to
train the model.

The model is trained using the Adam optimizer and
the Cross Entropy loss.

Usage:
    train.py <model> <train_data> <test_data> <input> <case> <mode>
            [--channel=<channel>] [--load=<model_name>]
            [--aggregate_labels] [--dropout=<dropout>]
            [--epochs=<epochs>] [--batch_size=<batch_size>]
            [--gradient_accumulation=<gradient_accumulation>]
            [--optimizer=<optimizer>] [--lr=<lr>]
            [--weight_decay=<weight_decay>] [--momentum=<momentum>]
            [--nesterov] [--scheduler=<scheduler>]
            [--loss=<loss>] [--early-stopping]
            [--patience=<patience>] [--min_delta=<min_delta>]
            [--verbose=<verbose>] [--seed=<seed>]
    train.py -h | --help

Options:
    -h --help                       Show this screen.
    <model>                         Model to use.
    <train_data>                    Name of the training data file.
    <test_data>                     Name of the test data file.
    <input>                         Type of data to use.
    <case>                          Case to use.
    <mode>                          Mode to use, possible values are total, standing, lying.
    --channel=<channel>             Channel to use [default: 1].
    --load=<model_name>             Load a model [default: None].
    --aggregate_labels              Aggregate the labels [default: False].
    --dropout=<dropout>             Dropout [default: 0.5].
    --epochs=<epochs>               Number of epochs [default: 100].
    --batch_size=<batch_size>       Batch size [default: 32].
    --gradient_accumulation=<gradient_accumulation>   Gradient accumulation steps [default: 1].
    --optimizer=<optimizer>         Optimizer [default: Adam].
    --lr=<lr>                       Learning rate [default: 0.001].
    --weight_decay=<weight_decay>   Weight decay [default: 0.0001].
    --momentum=<momentum>           Momentum [default: 0.9].
    --nesterov                      Nesterov [default: False].
    --scheduler=<scheduler>         Scheduler [default: None].
    --loss=<loss>                   Loss [default: CrossEntropyLoss].
    --early-stopping                Early stopping [default: False].
    --patience=<patience>           Patience [default: 10].
    --min_delta=<min_delta>         Minimum delta [default: 0.0001].
    --verbose=<verbose>             Verbose [default: 1].
    --seed=<seed>                   Seed [default: 42].

Example:
    python train.py CNN-MD train_mDoppler_2channels.h5 test_mDoppler_2channels.h5  mDoppler 3 --aggregate_labels --epochs=10  --weight_decay=0. --no-nesterov --dropout=0. --load

Current configuration:
    model: CNN-MD
    train_data: train_mDoppler_2channels_augmented.h5
    test_data: test_mDoppler_2channels.h5
    input: mDoppler
    case: 3
    dropout: 0.
    epochs: 100
    batch_size: 128
    weight_decay: 0.0
    aggregate_labels: True
    optimizer: Adam
    lr: 0.001
    loss: CrossEntropyLoss
    
    python train.py CNN-MD train_mDoppler_2channels_augmented.h5 test_mDoppler_2channels.h5 mDoppler 3 --dropout=0. --epochs=100 --batch_size=128 --weight_decay=0. --no-nesterov
"""

import torch
import numpy as np
from docopt import docopt
from models.classifier import model
from preprocessing.dataset import Dataset2Channels, Dataset1Channel
from utils.constants import MAPPING_LABELS_DICT, MAPPING_LYING_LABELS
from datetime import datetime
from torchvision import transforms

now = datetime.now().strftime("%Y%m%d")

def main(
    model_name:str,
    train_data,
    test_data,
    case:int,
    load:bool,
    in_channels:int,
    num_classes:int,
    dropout:float,
    epochs:int,
    batch_size:int,
    gradient_accumulation:int,
    optimizer:str,
    lr:float,
    weight_decay:float,
    momentum:float,
    nesterov:bool,
    scheduler:str,
    loss:str,
    labels_transform:callable,
    aggregate_labels:bool,
    early_stopping:bool,
    patience:int,
    min_delta:float,
    verbose:int,
    device:torch.device,
    mode:str,
    seed:int
):
    """
    Train the model, save the best model and
    save the training history

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
    labels_transform : callable
        Labels transform
    aggregate_labels : bool
        Aggregate the labels
    patience : int
        Patience
    min_delta : float
        Minimum delta
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
        device=device,
    )
    classifier.create_model(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=dropout
    )

    # Load the pre-trained model
    if load != "None":
        print(f'Loading model {load}')
        classifier.load_model(
            name=load,
        )
        
    # Create the DataLoaders
    classifier.create_DataLoaders(
        batch_size=batch_size,
        #num_workers=1
    )
    
    # Print the model summary
    classifier.summary(
        save=True,
        name=f'{model_name}_case{case}_summary.txt'
    )

    # Create the optimizer, loss function
    classifier.create_optimizer(
        optimizer=optimizer,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov
    )

    classifier.create_loss(
        loss=loss,
        use_weight=True,
        labels_transform=labels_transform,
    )
    
    del classifier.train_data, classifier.test_data

    # Create the early stopping
    if early_stopping:
        classifier.create_early_stopping(
            patience=patience,
            min_delta=min_delta,
            verbose=verbose,
            mode='max',
            baseline=.6,
            start_epoch=15,
            path=f'checkpoints/{model_name}__case_{case}_checkpoint.pt',
        )

    if scheduler != 'None':
        params = {
            'ReduceLROnPlateau': {
                'mode': 'min',
                'factor': 0.1,
                'patience': 10,
                'verbose': True,
                'threshold': 0.0001,
                'threshold_mode': 'rel',
                'cooldown': 0,
                'min_lr': 0,
                'eps': 1e-08
            },
            'StepLR': {
                'step_size': 10,
                'gamma': 0.1
            },
            'MultiStepLR': {
                'milestones': [10, 30, 60],
                'gamma': 0.1
            },
            'ExponentialLR': {
                'gamma': 0.1
            },
            'CosineAnnealingLR': {
                'T_max': 10,
                'eta_min': 0
            },
            'CyclicLR': {
                'base_lr': 0.001,
                'max_lr': 0.01,
                'step_size_up': 2000,
                'mode': 'triangular',
                'gamma': 1.0,
                'scale_fn': None,
                'scale_mode': 'cycle',
                'cycle_momentum': True,
                'base_momentum': 0.8,
                'max_momentum': 0.9,
                'last_epoch': -1
            },
        }
        classifier.create_scheduler(
            scheduler=scheduler,
            **params[scheduler]
        )

    # Train the model
    print('Training the model')
    classifier.train_model(
        epochs=epochs,
        accumulate_grad=gradient_accumulation,
    )

    # Plot the training history
    classifier.plot_history(
        save=True,
        show=False,
        save_csv=True
    )

    # Evaluate the model
    print('Evaluating the model')
    classifier.evaluate_model(
        save=True,
        aggregate=aggregate_labels,
        mode=mode
    )

    # Save the model trained
    print('Saving the model')
    classifier.save_trained_model(
        name=f'{model_name}_{now}_case{case}'
    )


if __name__ == '__main__':
    args = docopt(__doc__)

    # set seed
    seed = int(args['--seed'])
    torch.manual_seed(seed)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    load = args['--load']
    aggregate = bool(args['--aggregate_labels'])

    # set hyperparameters
    case = int(args['<case>'])
    mode = args['<mode>']
    dropout = float(args['--dropout'])
    epochs = int(args['--epochs'])
    batch_size = int(args['--batch_size'])
    optimizer = args['--optimizer']
    lr = float(args['--lr'])
    weight_decay = float(args['--weight_decay'])
    momentum = float(args['--momentum'])
    nesterov = bool(args['--nesterov'])
    loss = args['--loss']
    gradient_accumulation = int(args['--gradient_accumulation'])
    
    scheduler = args['--scheduler']
    early_stopping = bool(args['--early-stopping'])
    patience = int(args['--patience'])
    min_delta = float(args['--min_delta'])
    verbose = int(args['--verbose'])

    TYPE = args['<input>']
    
    if mode == 'lying':
        labels_transform = np.vectorize(
            lambda label: MAPPING_LYING_LABELS[label]
        )
    elif mode == aggregate:
        labels_transform = np.vectorize(
            lambda label: MAPPING_LABELS_DICT[label]
        )
    else:
        labels_transform = None

    if case == 0:

        features_transform = transforms.Compose([
            #lambda x: x[:, 9:-9],
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))
        ])

        train_data = Dataset1Channel(
            TYPE=TYPE,
            dirname='DATA_preprocessed',
            filename=args['<train_data>'],
            features_transform=features_transform,
            labels_transform=labels_transform,
            channel=int(args['--channel']),
        )
        
        test_data = Dataset1Channel(
            TYPE=TYPE,
            dirname='DATA_preprocessed',
            filename=args['<test_data>'],
            features_transform=features_transform,
            labels_transform=labels_transform,
            channel=int(args['--channel']),
        )
        in_channels = 1
        
    elif case == 1:
        
        features_transform = transforms.Compose([
            #lambda x: x[:, 9:-9],
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
        
        #### ATTENTION ####
        # The test data is the same as the train data
        # TODO: generate a new test set
        test_data = Dataset1Channel(
            TYPE=TYPE,
            dirname='DATA_preprocessed',
            filename=args['<train_data>'],
            features_transform=features_transform,
            labels_transform=labels_transform,
            channel=2,
        )
        in_channels = 1

    elif case == 2:
        # train and test on whole dataset
        # regardless of the radar, not suggested
        raise NotImplementedError('Case 2 not implemented yet')
    
    elif case == 3:
        features_transform = transforms.Compose([
            #lambda x: x[:, :, 9:-9],
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
    
    # shuffle data
    train_data.shuffle()
    test_data.shuffle()
    
    num_classes = len(np.unique(labels_transform(train_data.labels[:])))
    
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
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        optimizer=optimizer,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        loss=loss,
        labels_transform=labels_transform,
        aggregate_labels=aggregate,
        scheduler=scheduler,
        early_stopping=early_stopping,
        patience=patience,
        min_delta=min_delta,
        verbose=verbose,
        mode=mode,
        device=device,
        seed=seed
    )