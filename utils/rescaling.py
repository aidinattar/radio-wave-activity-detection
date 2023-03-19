'''
Functions for the main methods of rescaling data
'''

import numpy as np
from sklearn.decomposition import PCA

def normalize(data, **kwargs):
    '''
    Function to normalize the data

    Parameters
    ----------
    data : numpy.ndarray
        Data to normalize

    Returns
    -------
    numpy.ndarray
        Normalized data
    '''
    return (data - data.min()) / (data.max() - data.min())

def standardize(data, **kwargs):
    '''
    Function to standardize the data

    Parameters
    ----------
    data : numpy.ndarray
        Data to standardize

    Returns
    -------
    numpy.ndarray
        Standardized data
    '''
    return (data - data.mean()) / data.std()

def min_max(data, **kwargs):
    '''
    Function to apply min-max scaling to the data

    Parameters
    ----------
    data : numpy.ndarray
        Data to scale

    Returns
    -------
    numpy.ndarray
        Scaled data
    '''
    # Find the minimum value of the map
    data_min = np.min(data[data != 0])

    # Subtract the minimum value from the map
    data_norm = data - data_min

    # Find the maximum value of the normalized map
    data_max = np.max(data_norm)

    # Divide the normalized map by the maximum value
    return data_norm / data_max

def log_transformation(data, **kwargs):
    """
    Function to apply log transformation to the data

    Parameters:
    -----------
    data: numpy.ndarray
        Data to transform

    Returns:
    --------
    numpy.ndarray
        Transformed data
    """
    return np.log(data)

def power_transformation(data, power=1):
    """
    Function to apply power transformation to the data

    Parameters:
    -----------
    data: numpy.ndarray
        Data to transform

    power: float
        Power parameter to use in the transformation

    Returns:
    --------
    numpy.ndarray
        Transformed data
    """
    return np.power(data, power)

def binning(data, n_bins=10):
    """
    Function to apply binning to the data

    Parameters:
    -----------
    data: numpy.ndarray
        Data to bin

    n_bins: int
        Number of bins to use

    Returns:
    --------
    numpy.ndarray
        Binned data
    """
    bins = np.linspace(data.min(), data.max(), n_bins)
    return np.digitize(data, bins)

def apply_pca(data, n_components=2):
    """
    Function to apply PCA to the data

    Parameters:
    -----------
    data: numpy.ndarray
        Data to transform

    n_components: int
        Number of components to use in the transformation

    Returns:
    --------
    numpy.ndarray
        Transformed data
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)
