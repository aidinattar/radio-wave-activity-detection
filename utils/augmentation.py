'''
augmentation.py

This file contains the functions used to augment the data.
The functions are:
    - resample: resample the data to the default length
    - time_mask: apply a time mask to the data
    - doppler_mask: apply a doppler mask to the data
    + others...
'''
import torch
import numpy as np
from preprocessing.DataCutter import DataCutter
from preprocessing.DataProcess import DataProcess
from preprocessing.dataset import Dataset
from exceptions import OptionIsFalseError
from utils import cutting

from torch.utils.data import ConcatDataset

def cut_time(data: DataProcess,
             indices: list,
             loc: str = 'random',
             len_default: int = 40,
             n_samples: int = 5,
             type: str = 'mDoppler',
             **kwargs):
    '''
    Augment the data by cutting the time axis at random locations
    
    Parameters
    ----------
    data : DataProcess
        Data to augment.
    indices : list
        Indices of the data to augment. 
        The indices are the indices of the data in the
        training set.
    loc : str, optional
        Location of the cut. The default is 'random'.
        Possible values are:
            'random': cut at random locations
            'normal': cut at normal locations
    n_samples : int, optional
        Number of samples to take. The default is 5.
    type : str, optional
        Type of data to augment. The default is 'mDoppler'.
    **kwargs : dict
        Additional arguments to pass to the cut_time_rdn and cut_time_mDoppler functions.
    
    Raises
    ------
    ValueError
        If the type of data is invalid.
    
    Returns
    -------
    data : DataProcess
        Augmented data.
    '''
    if type=='rdn':
        cut_time_rdn(data, indices, loc=loc, len_default=len_default, n_samples=n_samples, **kwargs)
    elif type=='mDoppler':
        cut_time_mDoppler(data, indices, loc=loc, len_default=len_default, n_samples=n_samples, **kwargs)
    else:
        raise ValueError('Invalid type of data')

    return data

def cut_time_rdn(data: DataProcess,
                 indices: list,
                 loc: str = 'random',
                 len_default: int = 40,
                 n_samples: int = 5,
                 **kwargs):
    '''
    Augment the rdn data by cutting the time axis at random locations
    
    Parameters
    ----------
    data : DataProcess
        Data to augment.
    indices : list
        Indices of the data to augment.
        The indices are the indices of the data in the
        training set.
    loc : str, optional
        Location of the cut. The default is 'random'.
        Possible values are:
            'random': cut at random locations
            'normal': cut at normal locations
    len_default : int, optional
        Default length of the data. The default is 40.
    n_samples : int, optional
        Number of samples to take. The default is 5.
    
    Raises
    ------
    OptionIsFalseError
        If the option do_rdn is False.
    ValueError
        If the location is invalid.
    '''
    if not data.do_rdn:
        raise OptionIsFalseError('do_rdn')
    
    augmented_rdn_1 = []
    augmented_rdn_2 = []
    augmented_labels = []
    
    for i, (rdn_1, rdn_2, label) in enumerate(zip(data.rdn_1, data.rdn_2, data.labels)):
        if i in indices:
            for _ in range(n_samples):
                if loc == 'random':
                    augmented_rdn_1.append(cutting.random(rdn_1, len_default=len_default))
                    augmented_rdn_2.append(cutting.random(rdn_2, len_default=len_default))
                    augmented_labels.append(label)
                elif loc == 'normal':
                    augmented_rdn_1.append(cutting.normal(rdn_1, len_default=len_default, **kwargs))
                    augmented_rdn_2.append(cutting.normal(rdn_2, len_default=len_default, **kwargs))
                    augmented_labels.append(label)
                else:
                    raise ValueError('Invalid location')
    
    data.rdn_1 = augmented_rdn_1
    data.rdn_2 = augmented_rdn_2
    data.labels = augmented_labels
    
def cut_time_mDoppler(data: DataProcess,
                      indices: list,
                      loc: str = 'random',
                      len_default: int = 40,
                      n_samples: int = 5,
                      **kwargs):
    '''
    Augment the mDoppler data by cutting the time axis at random locations
    
    Parameters
    ----------
    data : DataProcess
        Data to augment.
    indices : list
        Indices of the data to augment.
        The indices are the indices of the data in the
        training set.
    loc : str, optional
        Location of the cut. The default is 'random'.
        Possible values are:
            'random': cut at random locations
            'normal': cut at normal locations
    len_default : int, optional
        Default length of the data. The default is 40.
    n_samples : int, optional
        Number of samples to take. The default is 5.
    
    Raises
    ------
    OptionIsFalseError
        If the option do_mDoppler is False.
    ValueError
        If the location is invalid.
    '''
    if not data.do_mDoppler:
        raise OptionIsFalseError('do_mDoppler')
    
    augmented_mDoppler_1 = []
    augmented_mDoppler_2 = []
    augmented_labels = []
    
    for i, (mDoppler_1, mDoppler_2, label) in enumerate(zip(data.mDoppler_1, data.mDoppler_2, data.labels)):
        if i in indices:
            for _ in range(n_samples):
                if loc == 'random':
                    augmented_mDoppler_1.append(cutting.random(mDoppler_1, len_default=len_default))
                    augmented_mDoppler_2.append(cutting.random(mDoppler_2, len_default=len_default))
                    augmented_labels.append(label)
                elif loc == 'normal':
                    augmented_mDoppler_1.append(cutting.normal(mDoppler_1, len_default=len_default, **kwargs))
                    augmented_mDoppler_2.append(cutting.normal(mDoppler_2, len_default=len_default, **kwargs))
                    augmented_labels.append(label)
                else:
                    raise ValueError('Invalid location')
                
    data.mDoppler_1 = augmented_mDoppler_1
    data.mDoppler_2 = augmented_mDoppler_2
    data.labels = augmented_labels


def resample(train_data: torch.utils.data.dataset.Subset,
             data_dir:str='DATA_preprocessed',
             data_file:str='data_cutted.npz',
             data_type:str='mDoppler',
             len_default: int = 40,
             n_samples: int = 5):
    '''
    Resample the array to the default length

    Parameters
    ----------
    train_data : np.ndarray
        Array to which samples will be added.
    data_dir : str, optional
        Directory of the data. The default is 'DATA_preprocessed'.
    data_file : str, optional
        File of the data. The default is 'data_cutted.npz'.
    data_type : str, optional
        Type of the data. The default is 'mDoppler'.
    len_default : int, optional
        Default length of the data. The default is 40.

    Returns
    -------
    array : np.ndarray
        Resampled array.
    '''
    data = DataCutter.from_file(data_dir, data_file)
    dp = DataProcess(data=data)
    dp = cut_time(dp, train_data.indices, loc='random', len_default=len_default, n_samples=n_samples, type=data_type)
    dp.padding(padding=len_default)
    dp.save(filename='data_resampled.npz')

    # Create a new dataset with additional data    
    new_data = Dataset(path='DATA_preprocessed', file='data_resampled.npz', type=data_type)

    # Update the Subset object to include the new data
    train_data.dataset = train_data.dataset.concat(new_data)
    train_data.indices = range(len(train_data.dataset))
    
    return train_data


def time_mask(train_data: torch.utils.data.dataset.Subset):
    pass


def doppler_mask(train_data: torch.utils.data.dataset.Subset):
    pass


def time_doppler_mask(train_data: torch.utils.data.dataset.Subset):
    pass