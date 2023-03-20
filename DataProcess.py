'''
DataProcess.py

This file contains the DataProcess class, which is used to process the data.
'''

import os
import numpy as np

from DataCutter import DataCutter

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

    def crop_rdn(self, start: int, stop: int):
        '''
        Crop the rdn data
        '''
        pass

    def crop_mDoppler(self, start: int, stop: int):
        '''
        Crop the mDoppler data
        '''
        pass