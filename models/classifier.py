'''
classifier.py
'''
from preprocessing.DataProcess import DataProcess
from exceptions.exceptions import OptionIsFalseError
from sklearn.model_selection import train_test_split

class model(object):
    '''
    Class to create the model
    '''

    def __init__(self, data: DataProcess):
        '''
        Constructor
        '''
        pass


    def create_model(self, model_type: str='CNN-MD'):
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


    def train_test_split(self, test_size: float=0.2):
        '''
        Split the data into training and testing data

        Parameters
        ----------
        test_size : float, optional
            Size of the test data. The default is 0.2.
        '''
        pass


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