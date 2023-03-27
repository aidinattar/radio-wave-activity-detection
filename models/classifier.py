'''
classifier.py
'''
import numpy as np
from torch.utils.data          import DataLoader, random_split
from torch.optim               import SGD, Adam
from torch.nn                  import BCELoss, BCEWithLogitsLoss
from preprocessing.dataset     import Dataset
from exceptions                import OptionIsFalseError, WorkToDoError
from sklearn.model_selection   import train_test_split
from cnn_rd                    import cnn_rd
from cnn_md                    import cnn_md
from tqdm                      import tqdm

class model(object):
    '''
    Class to create the model
    '''
    train_test_split_done = False
    model_created = False
    optimizer_created = False
    loss_created = False
    model_trained = False

    def __init__(self, data: Dataset, case: int=0, model_type: str='CNN-MD'):
        '''
        Constructor

        Parameters
        ----------
        data : Dataset
            Dataset object.
        case : int, optional
            Case to use. The default is 0.
            Possible values are:
                0: train and test separately on the radars
                1: train on one radar and test on the other
                2: train and test on all the data
        model_type : str, optional
            Type of model to create. Possible values are:
            'CNN-MD', 'CNN-RD'. The default is 'CNN-MD'.
        '''
        # Get the data
        self.data = data

        # Get the case
        self.case = case

        # Get the model type
        self.model_type = model_type


    # TODO: add also the option to use the validation set
    def train_test_split(self,
                         test_size: float=0.2,
                         random_state: int=42,
                         batch_size: int=32,
                         shuffle: bool=True,
                         num_workers: int=0):
        '''
        Split the data into training and testing data

        Parameters
        ----------
        test_size : float, optional
            Size of the test data. The default is 0.2.
        random_state : int, optional
            Random state. The default is 42.
        batch_size : int, optional
            Batch size. The default is 32.
        shuffle : bool, optional
            Shuffle the data. The default is True.
        num_workers : int, optional
            Number of workers. The default is 0.
        '''
        # Split the data
        if self.case == 0:
            train_size = int((1-test_size) * len(self.data))
            test_size = len(self.data) - train_size
            self.train_data, self.test_data = random_split(self.data, [train_size, test_size])
        elif self.case == 1:
            # TODO: split the data into training and testing data
            pass
        elif self.case == 2:
            # TODO: split the data into training and testing data
            pass
        else:
            raise ValueError('Invalid case')

        # Create the data loaders
        self.train_loader = DataLoader(self.train_data,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        num_workers=num_workers)
        self.test_loader = DataLoader(self.test_data,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        num_workers=num_workers)

        # Set the flag
        self.train_test_split_done = True


    def create_model(self):
        '''
        Create the model
        '''
        # call cnn_rd or cnn_md class
        if self.model_type == 'CNN-MD':
            self.model = cnn_md(
                # params here
            )
        elif self.model_type == 'CNN-RD':
            self.model = cnn_rd(
                # params here
            )
        else:
            raise ValueError('Invalid model type')

        # Set the flag
        self.model_created = True


    def create_optimizer(self, optimizer: str='SGD', lr: float=0.01, momentum: float=0.9):
        '''
        Create the optimizer

        Parameters
        ----------
        optimizer : str, optional
            Optimizer to use. The default is 'SGD'.
            Possible values are:
                'SGD': Stochastic Gradient Descent
                'Adam': Adam optimizer
        lr : float, optional
            Learning rate. The default is 0.01.
        momentum : float, optional
            Momentum. The default is 0.9.
        '''
        if optimizer == 'SGD':
            self.optimizer = SGD(self.model.parameters(), lr=lr, momentum=momentum)
        elif optimizer == 'Adam':
            self.optimizer = Adam(self.model.parameters(), lr=lr)
        else:
            raise ValueError('Invalid optimizer')

        # Set the flag
        self.optimizer_created = True


    def create_loss(self, loss: str='BCELoss'):
        '''
        Create the loss function

        Parameters
        ----------
        loss : str, optional
            Loss function to use. The default is 'BCELoss'.
            Possible values are:
                'BCELoss': Binary Cross Entropy Loss
                'BCEWithLogitsLoss': Binary Cross Entropy with Logits Loss
        '''
        if loss == 'BCELoss':
            self.loss = BCELoss()
        elif loss == 'BCEWithLogitsLoss':
            self.loss = BCEWithLogitsLoss()
        else:
            raise ValueError('Invalid loss function')

        # Set the flag
        self.loss_created = True


    def train_model(self,
                    epochs: int=10,
                    device: str='cuda',
                    checkpoint: bool=False,
                    checkpoint_path: str='checkpoint.pth',
                    ):
        '''
        Train the model

        Parameters
        ----------
        epochs : int, optional
            Number of epochs. The default is 10.
        device : str, optional
            Device to use. The default is 'cuda'.
            Possible values are:
                'cpu': CPU
                'cuda': GPU
        checkpoint : bool, optional
            Save the model after every epoch. The default is False.
        checkpoint_path : str, optional
            Path to save the model. The default is 'checkpoint.pth'.
        '''
        if not self.train_test_split_done:
            raise WorkToDoError('train_test_split_done')
        if not self.model_created:
            raise WorkToDoError('model_created')
        if not self.optimizer_created:
            raise WorkToDoError('optimizer_created')
        if not self.loss_created:
            raise WorkToDoError('loss_created')

        # Use GPU
        self.model = self.model.to(device)

        # Train the model
        for epoch in tqdm(range(epochs)):
            pass

        self.model_trained = True


    def evaluate_model(self,
                       do_cm: bool=True,
                       do_acc: bool=True,
                       do_prec: bool=True,
                       do_rec: bool=True,
                       do_f1: bool=True,
                       do_roc: bool=True,
                       do_auc: bool=True,
                       ):
        '''
        Evaluate the model.

        Confusion matrix, accuracy, precision, recall, f1-score,
        ROC curve, AUC, etc.
        '''
        if not self.model_trained:
            raise WorkToDoError('model_trained')

        # Confusion matrix
        if do_cm:
            pass

        # Accuracy
        if do_acc:
            pass

        # Precision
        if do_prec:
            pass

        # Recall
        if do_rec:
            pass

        # F1-score
        if do_f1:
            pass

        # ROC curve
        if do_roc:
            pass

        # AUC
        if do_auc:
            pass


    def save_model(self):
        '''
        Save the model
        '''
        pass


    def load_model(self):
        '''
        Load the model
        '''
        pass