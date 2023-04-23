'''
train.py

This file contains the train function, which is used to train the model.

The model is trained using the Adam optimizer and the Cross Entropy loss.

Usage:
    train.py <model> <data> <input> <case> (--load|--no-load) [--augment|--no-augment] [--epochs=<epochs>] [--batch_size=<batch_size>] [--optimizer=<optimizer>] [--lr=<lr>] [--weight_decay=<weight_decay>] [--momentum=<momentum>] [--nesterov=<nesterov>] [--loss=<loss>] [--patience=<patience>] [--min_delta=<min_delta>] [--factor=<factor>] [--verbose=<verbose>] [--seed=<seed>]
    train.py -h | --help

Options:
    -h --help                       Show this screen.
    --augment                       Augment the data [default: False].
    --epochs=<epochs>               Number of epochs [default: 100].
    --batch_size=<batch_size>       Batch size [default: 32].
    --optimizer=<optimizer>         Optimizer [default: Adam].
    --lr=<lr>                       Learning rate [default: 0.001].
    --weight_decay=<weight_decay>   Weight decay [default: 0.0001].
    --momentum=<momentum>           Momentum [default: 0.9].
    --nesterov=<nesterov>           Nesterov [default: False].
    --loss=<loss>                   Loss [default: CrossEntropyLoss].
    --patience=<patience>           Patience [default: 10].
    --min_delta=<min_delta>         Minimum delta [default: 0.0001].
    --factor=<factor>               Factor [default: 0.1].
    --verbose=<verbose>             Verbose [default: 1].
    --seed=<seed>                   Seed [default: 42].

Example:
    python train.py CNN-MD data_processed.npz mDoppler 2 --load --augment --epochs=1500 --batch_size=32 --optimizer=Adam --lr=0.001 --weight_decay=0.0001 --momentum=0.9 --nesterov=True --loss=Adam --patience=10 --min_delta=0.0001 --factor=0.1 --verbose=1 --seed=42
'''

# TODO:
# Add parameters to the docstring
# Add the parameters not used to the class
# Check the correctness of the code

import torch
from docopt                    import docopt
from models.classifier         import model
from preprocessing.dataset     import Dataset
from datetime import datetime

now = datetime.now().strftime("%Y%m%d")

def main(model_name:str, data:Dataset, case, load, augment, epochs, batch_size, optimizer, lr, weight_decay, momentum, nesterov, loss, patience, min_delta, factor, verbose, device):
    '''
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
    augment : bool
        Augment the data
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
    '''
    # Create the model object
    classifier = model(data=data, case=case, model_type=model_name)
    classifier.create_model()

    # Load the pre-trained model
    if load:
        classifier.load_model(name='checkpoint', path='checkpoints')

    # Split the data into training and validation sets
    classifier.train_test_split(test_size=.2)
    
    # Augment the data
    if augment:
        classifier.augmentation(method=['resample'])
        
    # Create the DataLoaders
    classifier.create_DataLoaders(batch_size=batch_size)
    
    # Print the model summary
    classifier.summary(save=True, name=f'{model_name}_case{case}_summary.txt')

    # Create the optimizer, loss function
    classifier.create_optimizer(optimizer=optimizer, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
    classifier.create_loss(loss=loss)

    # Train the model
    classifier.train_model(epochs=epochs, checkpoint=True, checkpoint_path=f'{model_name}__case_{case}_checkpoint.pt')

    # Plot the training history
    classifier.plot_history(save=True)

    # Evaluate the model
    classifier.evaluate_model(do_roc_auc=False)

    # Save the model trained
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

    # set hyperparameters
    case = int(args['<case>'])
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

    # load data
    data = Dataset(
        path='DATA_preprocessed',
        file=args['<data>'],
        type=args['<input>'],
        transform=None
    )

    # load model
    model_name = args['<model>']

    main(model_name=model_name,
         data=data,
         case=case,
         load=load,
         augment=augment,
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
         device=device)