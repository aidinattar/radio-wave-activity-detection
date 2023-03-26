'''
DataProcess.py

This file contains the DataProcess class, which is used to process the data.
'''

import os
import numpy as np

from preprocessing.DataCutter import DataCutter
from exceptions.exceptions import OptionIsFalseError

class DataProcess(object):
    '''
    Class to process the data
    '''

    def __init__(self, data: DataCutter):
        '''
        Constructor
        '''
        self.data = data
        self.do_rdn = self.data.data.do_rdn
        self.do_mDoppler = self.data.data.do_mDoppler

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

    def padding_rdn(self, padding: int, mode: str='constant', **kwargs):
        '''
        Pad the rdn data.
        Check if the action is at least 40 bins long, if not, pad the data.

        Parameters
        ----------
        padding : int
            Number of bins to pad.
        mode : str, optional
            Mode to use to pad the data. The default is 'constant'.
        **kwargs : TYPE
            Keyword arguments to pass to the pad function.
        '''
        for i, (rdn_1, rdn_2) in enumerate(zip(self.data.signal_rdn_1, self.data.signal_rdn_2)):
            if rdn_1.shape[0] < 40:
                self.rdn_1 = np.pad(
                    array=rdn_1,
                    pad_width=((0, padding - rdn_1.shape[0]), (0, 0)),
                    mode=mode,
                    **kwargs
                )
            else:
                self.rdn_1 = rdn_1
            if rdn_2.shape[0] < 40:
                self.rdn_2 = np.pad(
                    array=rdn_2,
                    pad_width=((0, padding - rdn_2.shape[0]), (0, 0)),
                    mode=mode,
                    **kwargs
                )
            else:
                self.rdn_2 = rdn_2

    def padding_mDoppler(self, padding:int, mode: str='constant', len_default:int=40, **kwargs):
        '''
        Pad the mDoppler data.
        Check if the action is at least 40 bins long, if not, pad the data.

        Parameters
        ----------
        padding : int
            Number of bins to pad.
        mode : str, optional
            Mode to use to pad the data. The default is 'constant'.
        len_default : int, optional
            Default length of the action. The default is 40.
        **kwargs : TYPE
            Keyword arguments to pass to the pad function.
        '''


        for i, (mDoppler_1, mDoppler_2) in enumerate(zip(self.data.signal_mDoppler_1, self.data.signal_mDoppler_2)):
            if mDoppler_1.shape[0] < len_default:
                self.mDoppler_1 = np.pad(
                    array=mDoppler_1,
                    pad_width=((0, padding - mDoppler_1.shape[0]), (0, 0)),
                    mode=mode,
                    **kwargs
                )
            else:
                self.mDoppler_1 = mDoppler_1
            if mDoppler_2.shape[0] < len_default:
                self.mDoppler_2 = np.pad(
                    array=mDoppler_2,
                    pad_width=((0, padding - mDoppler_2.shape[0]), (0, 0)),
                    mode=mode,
                    **kwargs
                )
            else:
                self.mDoppler_2 = mDoppler_2


    def cut_time(self, loc:str='random', len_default:int=40):
        '''
        Cut the data in time

        Parameters
        ----------
        loc : str, optional
            Location of the cut. Possible values are:
            'center', 'start', 'end', 'random', 'normal'.
            The default is 'random'.
        '''
        if self.do_rdn:
            self.cut_time_rdn(loc=loc, len_default=len_default)
        if self.do_mDoppler:
            self.cut_time_mDoppler(loc=loc, len_default=len_default)


    def cut_time_rdn(self, loc:int='random', len_default:int=40):
        '''
        Cut the rdn data in time

        Parameters
        ----------
        loc : str, optional
            Location of the cut. Possible values are:
            'center', 'start', 'end', 'random', 'normal'.
            The default is 'random'.
        len_default : int, optional
            Default length of the action. The default is 40.

        Raises
        ------
        OptionIsFalseError
            If the option do_rdn is False.
        ValueError
            If the location is not valid.
        '''
        if not self.do_rdn:
            raise OptionIsFalseError("do_rdn")

        for i, (rdn_1, rdn_2) in enumerate(zip(self.data.signal_rdn_1, self.data.signal_rdn_2)):
            if loc == 'center':
                self.rdn_1 = rdn_1[rdn_1.shape[0]//2-len_default//2:rdn_1.shape[0]//2+len_default//2, :]
                self.rdn_2 = rdn_2[rdn_2.shape[0]//2-len_default//2:rdn_2.shape[0]//2+len_default//2, :]
            elif loc == 'start':
                self.rdn_1 = rdn_1[:len_default, :]
                self.rdn_2 = rdn_2[:len_default, :]
            elif loc == 'end':
                self.rdn_1 = rdn_1[-len_default:, :]
                self.rdn_2 = rdn_2[-len_default:, :]
            elif loc == 'random':
                pass
            elif loc == 'normal':
                pass
            else:
                raise ValueError("Invalid location")

    def cut_time_mDoppler(self, loc:int='random', len_default:int=40):
        '''
        Cut the mDoppler data in time

        Parameters
        ----------
        loc : str, optional
            Location of the cut. Possible values are:
            'center', 'start', 'end', 'random', 'normal'.
            The default is 'random'.
        len_default : int, optional
            Default length of the action. The default is 40.

        Raises
        ------
        OptionIsFalseError
            If do_mDoppler is False.
        ValueError
            If loc is not a valid value.
        '''
        if not self.do_mDoppler:
            raise OptionIsFalseError("do_mDoppler")

        for i, (mDoppler_1, mDoppler_2) in enumerate(zip(self.data.signal_mDoppler_1, self.data.signal_mDoppler_2)):
            if loc == 'center':
                self.mDoppler_1 = mDoppler_1[mDoppler_1.shape[0]//2-len_default//2:mDoppler_1.shape[0]//2+len_default//2, :]
                self.mDoppler_2 = mDoppler_2[mDoppler_2.shape[0]//2-len_default//2:mDoppler_2.shape[0]//2+len_default//2, :]
            elif loc == 'start':
                self.mDoppler_1 = mDoppler_1[:len_default, :]
                self.mDoppler_2 = mDoppler_2[:len_default, :]
            elif loc == 'end':
                self.mDoppler_1 = mDoppler_1[-len_default:, :]
                self.mDoppler_2 = mDoppler_2[-len_default:, :]
            elif loc == 'random':
                pass
            elif loc == 'normal':
                pass
            else:
                raise ValueError("Invalid location")

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