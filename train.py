'''
train.py

This file contains the train function, which is used to train the model.

The model is trained using the Adam optimizer and the Cross Entropy loss.

Usage:
    train.py <model> <data> <output> [--epochs=<epochs>] [--batch_size=<batch_size>] [--lr=<lr>] [--weight_decay=<weight_decay>] [--momentum=<momentum>] [--nesterov=<nesterov>] [--patience=<patience>] [--min_delta=<min_delta>] [--factor=<factor>] [--verbose=<verbose>] [--seed=<seed>]
    train.py -h | --help

Options:
    -h --help                   Show this screen.
    --epochs=<epochs>           Number of epochs [default: 100].
    --batch_size=<batch_size>   Batch size [default: 32].
    --lr=<lr>                   Learning rate [default: 0.001].
    --weight_decay=<weight_decay>   Weight decay [default: 0.0001].
    --momentum=<momentum>       Momentum [default: 0.9].
    --nesterov=<nesterov>       Nesterov [default: True].
    --patience=<patience>       Patience [default: 10].
    --min_delta=<min_delta>     Minimum delta [default: 0.0001].
    --factor=<factor>           Factor [default: 0.1].
    --verbose=<verbose>         Verbose [default: 1].
    --seed=<seed>               Seed [default: 42].
'''
from docopt import docopt
import torch
from DataProcess import DataProcess

def main(data:DataProcess, output, epochs, batch_size, lr, weight_decay, momentum, nesterov, patience, min_delta, factor, verbose, device):
    pass

if __name__ == '__main__':
    args = docopt(__doc__)

    # set seed
    seed = int(args['--seed'])
    torch.manual_seed(seed)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set hyperparameters
    epochs = int(args['--epochs'])
    batch_size = int(args['--batch_size'])
    lr = float(args['--lr'])
    weight_decay = float(args['--weight_decay'])
    momentum = float(args['--momentum'])
    nesterov = bool(args['--nesterov'])
    patience = int(args['--patience'])
    min_delta = float(args['--min_delta'])
    factor = float(args['--factor'])
    verbose = int(args['--verbose'])

    # load data
    data = DataProcess(args['<data>'])

    # load model
    model = args['<model>']

    main(model, data, args['<output>'], epochs, batch_size, lr, weight_decay, momentum, nesterov, patience, min_delta, factor, verbose, device)