from preprocessing import DataCutter, DataReader
import numpy as np

def resample(train_data: np.ndarray,
             data_dir:str='DATA_preprocessed',
             data_file:str='data_cutted.npz',
             data_type:str='mDoppler',
             len_default: int = 40):
    '''
    Resample the array to the default length

    Parameters
    ----------
    train_data : np.ndarray
        Array to which samples will be added.

    Returns
    -------
    array : np.ndarray
        Resampled array.
    '''
    dr = DataReader(0, 0)
    data = DataCutter(data_dir, data_file)