'''
cutting.py

Classes and functions for cutting the data.
'''

import numpy as np

def random(array: np.ndarray, len_default: int = 40):
    '''
    Cut the array in time at a random location

    Parameters
    ----------
    array : np.ndarray
        Array to cut.
    len_default : int, optional
        Default length of the action. The default is 40.

    Returns
    -------
    array : np.ndarray
        Cut array.
    '''
    if array.shape[0] <= len_default:
        return array
    else:
        # select len_default bins randomly and 
        # take the data sorted by time
        random_bins = np.random.choice(
            a=array.shape[0],
            size=len_default,
            replace=False
        )
        random_bins.sort()
        return array[random_bins]


def gaussian(array: np.ndarray,
             loc: float=20,
             scale: float=5):
    '''
    Get the probability of a value x according to a gaussian distribution
    
    Parameters
    ----------
    array : np.ndarray
        array of values.
    loc : float, optional
        Mean of the gaussian distribution. The default is 20.
    scale : float, optional
        Standard deviation of the gaussian distribution. The default is 5.
    '''
    return np.exp(-np.power(array - loc, 2.) / (2 * np.power(scale, 2.))) / (scale * np.sqrt(2 * np.pi))


def normal(array: np.ndarray,
           len_default:int=40,
           mean:int=20,
           std:int=5):
    '''
    Cut the array in time at a normal location

    Parameters
    ----------
    array : np.ndarray
        Array to cut.
    len_default : int, optional
        Default length of the action. The default is 40.

    Returns
    -------
    array : np.ndarray
        Cut array.
    '''
    if array.shape[0] <= len_default:
        return array
    else:
        # select len_default bins according to a normal
        # distribution and take the data sorted by time
        normal_distrib = gaussian(
            array=np.arange(len(array)),
            loc=len(array)//2,
            scale=std,
        )

        normal_bins = np.random.choice(
            a=array.shape[0],
            size=len_default,
            replace=False,
            p=(normal_distrib / normal_distrib.sum())
        )

        normal_bins.sort()
        return array[normal_bins]