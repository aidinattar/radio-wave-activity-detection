'''
DataProcess.py

This file contains the DataProcess class, which is used to process the data.
'''

import os
import numpy as np

from preprocessing.DataCutter import DataCutter

class DataProcess(object):
    '''
    Class to process the data
    '''

    def __init__(self, data: DataCutter):
        '''
        Constructor
        '''
        self.data = data

    def agglomerate_rdn(self, method: str='max'):
        '''
        Agglomerate the rdn data on the time axis.

        Parameters
        ----------
        method : str, optional
            Method to use to agglomerate the data. Possible values are:
            'max', 'min', 'mean', 'median', 'sum', 'std', 'var'. The default is 'max'.
        '''

        # create dictionary of methods
        methods = {
            'max': np.max,
            'min': np.min,
            'mean': np.mean,
            'median': np.median,
            'sum': np.sum,
            'std': np.std,
            'var': np.var
        }

        if method not in methods:
            raise ValueError("Invalid method")

        for i, (rdn_1, rdn_2) in enumerate(zip(self.data.signal_rdn_1, self.data.signal_rdn_2)):
            self.rdn_1 = self.rdn[i].max(axis=0)
            self.rdn_2 = self.rdn[i].max(axis=0)

    def padding_rdn(self, padding: int):
        '''
        Pad the rdn data.
        Check if the action is at least 40 bins long.
        '''
        pass

    def padding_mDoppler(self, padding: int):
        '''
        Pad the mDoppler data
        '''
        pass


    def cut_time(self, loc='random'):
        '''
        Cut the data in time

        Parameters
        ----------
        loc : str, optional
            Location of the cut. Possible values are:
            'center', 'start', 'end', 'random'.
            The default is 'random'.
        '''
        pass


    def augmentation(self):
        '''
        Augment the data
        '''
        pass


    def save(self, path:str='DATA_preprocessed', filename:str='data_processed.npz'):
        '''
        Save the data.
        '''

        # Create the path
        path = os.path.join(path, filename)

        # Save the data
        np.savez(path, ###### Arrays here)


    def load(self, path:str='DATA_preprocessed', filename:str='data_processed.npz'):
        '''
        Load the data.
        '''

        # Create the path
        path = os.path.join(path, filename)

        # Load the data
        data = np.load(path)

        ###### Load the arrays here
        pass