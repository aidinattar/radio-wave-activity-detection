'''
classifier.py
'''
import numpy as np
from torch.utils.data          import DataLoader, random_split
from preprocessing.dataset     import Dataset
from exceptions                import OptionIsFalseError
from sklearn.model_selection   import train_test_split
from cnn_rd                    import CNN_RD
from cnn_md                    import CNN_MD

class model(object):
    '''
    Class to create the model
    '''

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


    def create_model(self):
        '''
        Create the model

        Parameters
        ----------
        model_type : str, optional
            Type of model to create. Possible values are:
            'CNN-MD', 'CNN-RD'. The default is 'CNN-MD'.
        '''
        # call cnn_rd or cnn_md class
        pass


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


    def train_model(self):
        '''
        Train the model
        '''
        pass


    def evaluate_model(self):
        '''
        Evaluate the model.

        Confusion matrix, accuracy, precision, recall, f1-score,
        ROC curve, AUC, etc.
        '''
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