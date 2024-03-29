"""
cutting.py

Classes and functions for cutting the data.
"""

import numpy as np
import sys
from scipy.ndimage import convolve
from scipy.signal import convolve2d

sys.setrecursionlimit(10000)


def random(array: np.ndarray, len_default: int = 40):
    """
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
    """
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
    """
    Get the probability of a value x according to a gaussian distribution
    
    Parameters
    ----------
    array : np.ndarray
        array of values.
    loc : float, optional
        Mean of the gaussian distribution. The default is 20.
    scale : float, optional
        Standard deviation of the gaussian distribution. The default is 5.
    """
    return np.exp(-np.power(array - loc, 2.) / (2 * np.power(scale, 2.))) / (scale * np.sqrt(2 * np.pi))


def normal(array: np.ndarray,
           len_default:int=40,
           mean:int=20,
           std:int=5):
    """
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
    """
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
   

def centered_moving_average(
    arr: np.ndarray,
    window_size: int=5
)->np.ndarray:
    """
    Calculate the centered moving average of an array
    
    Parameters
    ----------
    arr : np.ndarray
        Array to calculate the moving average.
    window_size : int, optional
        Size of the window. The default is 5.
        
    Returns
    -------
    moving_avg : np.ndarray
        Array with the moving average.
    """
    # Define the weights for the moving average window
    weights = np.ones((window_size,)*len(arr.shape)) / window_size**len(arr.shape)
    
    if len(arr.shape) == 1:

        # Use np.convolve() to calculate the centered moving average
        moving_avg = np.convolve(arr, weights, mode='valid')

        # Calculate the number of values to be padded on each side
        pad_width = window_size // 2

        # Pad the output array with edge values
        padded_avg = np.pad(moving_avg, pad_width, mode='edge')

        return padded_avg
    
    elif len(arr.shape) == 2:
        
        moving_average = convolve2d(arr, weights, mode='valid')
        
        pad_width = window_size // 2
        
        padded_avg = np.pad(moving_average, pad_width, mode='edge')
        
        return padded_avg
   
    
def threshold_method_(array: np.ndarray,
                     len_default:int=40,
                     loc:str='start',
                     threshold:float=.5,
                     recursions:int=0):
    """
    Cut the data according to a threshold
    If the threshold is not reached, the threshold
    is lowered by 10% recursively until it is reached
    
    Parameters
    ----------
    array : np.ndarray
        Array to cut.
    len_default : int, optional
        Default length of the action. The default is 40.
    loc : str, optional
        Location of the threshold. The default is 'start'.
        Possible values are 'start', 'end' and 'center'.
    threshold : float, optional
        Threshold to cut. The default is .5.
    recursions : int, optional
        Number of recursions. The default is 0.
    """
    max_recursions = 999
    flag_return = False
    
    if array.shape[0] <= len_default:
        return array
    
    else:
        array_copy = array.copy()
        means = array_copy.mean(axis=1)
        means = centered_moving_average(arr=means, window_size=10)
        try:
            first = np.where(means > threshold)[0][0]
            last = np.where(means > threshold)[0][-1]
        except IndexError:
            if recursions < max_recursions:
                return threshold_method(array=array,
                                        len_default=len_default,
                                        loc=loc,
                                        threshold=threshold*.99,
                                        recursions=recursions+1)
            else:
                flag_return = True
        
        if loc == 'start':
            if 'first' in locals():
                if first + len_default < array.shape[0]:
                    return array[first:first+len_default]
                else:
                    if not flag_return and recursions < max_recursions:
                        return threshold_method(array=array,
                                                len_default=len_default,
                                                loc=loc,
                                                threshold=threshold*.99,
                                                recursions=recursions+1)
                    else:
                        if np.where(means > threshold)[0].size == 0:
                            return array[:len_default]
                        else:
                            return array[-len_default:]
            else:
                if not flag_return and recursions < max_recursions:
                    return threshold_method(array=array,
                                            len_default=len_default,
                                            loc=loc,
                                            threshold=threshold*.99,
                                            recursions=recursions+1)
                else:
                    if np.where(means > threshold)[0].size == 0:
                        return array[:len_default]
                    else:
                        return array[-len_default:]
        elif loc == 'end':
            if 'first' in locals():
                if last - len_default > 0:
                    return array[last-len_default:last]
                else:
                    if not flag_return and recursions < max_recursions:
                        return threshold_method(array=array,
                                                len_default=len_default,
                                                loc=loc,
                                                threshold=threshold*.99,
                                                recursions=recursions+1)
                    else:
                        if np.where(means > threshold)[0].size == 0:
                            return array[-len_default:]
                        else:
                            return array[:len_default]
            else:
                if not flag_return:
                    return threshold_method(array=array,
                                            len_default=len_default,
                                            loc=loc,
                                            threshold=threshold*.99,
                                            recursions=recursions+1)
                else:
                    if np.where(means > threshold)[0].size == 0:
                        return array[-len_default:]
                    else:
                        return array[:len_default]
        elif loc == 'center':
            if 'first' in locals() and 'last' in locals():
                if (last - first) > len_default:
                    center = (last + first) // 2
                    return array[center-len_default//2:center+len_default//2]
                elif (last - first) == len_default:
                    return array[first:last]
                else:
                    if not flag_return:
                        return threshold_method(array=array,
                                                len_default=len_default,
                                                loc=loc,
                                                threshold=threshold*.99,
                                                recursions=recursions+1)
                    else:
                        if np.where(means > threshold)[0].size == 0:
                            center = array.shape[0] // 2
                        else:
                            center = (last + first) // 2
                        return array[center-len_default//2:center+len_default//2]
            else:
                if not flag_return:
                    return threshold_method(array=array,
                                            len_default=len_default,
                                            loc=loc,
                                            threshold=threshold*.99,
                                            recursions=recursions+1)
                else:
                    if np.where(means > threshold)[0].size == 0:
                        center = array.shape[0] // 2
                    else:
                        center = (last + first) // 2
                    return array[center-len_default//2:center+len_default//2]
        else:
            raise ValueError('loc must be "start", "end" or "center"')


def moving_average(
    arr: np.ndarray,
    window_size: int = 5
) -> np.ndarray:
    """
    Calculate the centered moving average of an array.

    Parameters
    ----------
    arr : np.ndarray
        Array to calculate the moving average.
    window_size : int, optional
        Size of the window. The default is 5.

    Returns
    -------
    moving_avg : np.ndarray
        Array with the moving average.
    """
    if arr.ndim == 2:
        weights = np.ones((window_size, window_size)) / (window_size ** 2)
        moving_avg = convolve(arr, weights, mode='constant', cval=0.0)
    elif arr.ndim == 3:
        weights = np.ones((window_size, window_size, window_size)) / (window_size ** 3)
        moving_avg = convolve(arr, weights, mode='constant', cval=0.0)
    else:
        raise ValueError("Input array must be 2D or 3D.")

    return moving_avg

        
        
def threshold_method(array: np.ndarray,
                     len_default: int = 40,
                     loc: str = 'start',
                     threshold: float = 1,
                     recursions: int = 0):
    """
    Cut the data according to a threshold based on integrals.
    If the threshold is not reached, the threshold
    is lowered by 10% recursively until it is reached.
    
    Parameters
    ----------
    array : np.ndarray
        Array to cut.
    len_default : int, optional
        Default length of the action. The default is 40.
    loc : str, optional
        Location of the threshold. The default is 'start'.
        Possible values are 'start', 'end' and 'center'.
    threshold : float, optional
        Threshold to cut. The default is 0.5.
    recursions : int, optional
        Number of recursions. The default is 0.
    """
    max_recursions = 999
    flag_return = False
    
    if array.shape[0] <= len_default:
        return array
    
    else:
        array_copy = array.copy()
        integrals = np.cumsum(array_copy, axis=1)
        integrals = moving_average(arr=integrals, window_size=10)
        
        try:
            first = np.where(integrals > threshold * integrals.max())[0][0]
            last = np.where(integrals > threshold * integrals.max())[0][-1]
        except IndexError:
            if recursions < max_recursions:
                return threshold_method(array=array,
                                        len_default=len_default,
                                        loc=loc,
                                        threshold=threshold * 0.99,
                                        recursions=recursions + 1)
            else:
                flag_return = True
        
        if loc == 'start':
            if 'first' in locals():
                if first + len_default < array.shape[0]:
                    return array[first:first + len_default]
                else:
                    if not flag_return and recursions < max_recursions:
                        return threshold_method(array=array,
                                                len_default=len_default,
                                                loc=loc,
                                                threshold=threshold * 0.99,
                                                recursions=recursions + 1)
                    else:
                        if np.where(integrals > threshold * integrals.max())[0].size == 0:
                            return array[:len_default]
                        else:
                            return array[-len_default:]
            else:
                if not flag_return and recursions < max_recursions:
                    return threshold_method(array=array,
                                            len_default=len_default,
                                            loc=loc,
                                            threshold=threshold * 0.99,
                                            recursions=recursions + 1)
                else:
                    if np.where(integrals > threshold * integrals.max())[0].size == 0:
                        return array[:len_default]
                    else:
                        return array[-len_default:]
        elif loc == 'end':
            if 'first' in locals():
                if last - len_default > 0:
                    return array[last - len_default:last]
                else:
                    if not flag_return and recursions < max_recursions:
                        return threshold_method(array=array,
                                                len_default=len_default,
                                                loc=loc,
                                                threshold=threshold * 0.99,
                                                recursions=recursions + 1)
                    else:
                        if np.where(integrals > threshold * integrals.max())[0].size == 0:
                            return array[-len_default:]
                        else:
                            return array[:len_default]
            else:
                if not flag_return:
                    return threshold_method(array=array,
                                            len_default=len_default,
                                            loc=loc,
                                            threshold=threshold * 0.99,
                                            recursions=recursions + 1)
                else:
                    if np.where(integrals > threshold * integrals.max())[0].size == 0:
                        return array[-len_default:]
                    else:
                        return array[:len_default]
        elif loc == 'center':
            if 'first' in locals() and 'last' in locals():
                if (last - first) > len_default:
                    center = (last + first) // 2
                    return array[center - len_default // 2:center + len_default // 2]
                elif (last - first) == len_default:
                    return array[first:last]
                else:
                    if not flag_return:
                        return threshold_method(array=array,
                                                len_default=len_default,
                                                loc=loc,
                                                threshold=threshold * 0.99,
                                                recursions=recursions + 1)
                    else:
                        if np.where(integrals > threshold * integrals.max())[0].size == 0:
                            center = array.shape[0] // 2
                        else:
                            center = (last + first) // 2
                        return array[center - len_default // 2:center + len_default // 2]
            else:
                if not flag_return:
                    return threshold_method(array=array,
                                            len_default=len_default,
                                            loc=loc,
                                            threshold=threshold * 0.99,
                                            recursions=recursions + 1)
                else:
                    if np.where(integrals > threshold * integrals.max())[0].size == 0:
                        center = array.shape[0] // 2
                    else:
                        center = (last + first) // 2
                    return array[center - len_default // 2:center + len_default // 2]
        else:
            raise ValueError('loc must be "start", "end" or "center"')


def find_highest_integral_frames(
    array: np.ndarray,
    len_default: int = 40
) -> np.ndarray:
    """
    Find the frames with the highest integral values.
    
    Parameters
    ----------
    array : np.ndarray
        Array to cut.
    len_default : int, optional
        Default length of the action. The default is 40.
        
    Returns
    -------
    array : np.ndarray
        Array with the highest integral values.
    """
    if array.ndim == 2:
        return find_highest_integral_frames_2d(array, len_default)
    elif array.ndim == 3:
        return find_highest_integral_frames_3d(array, len_default)
    else:
        raise ValueError("Array dimension should be 2 or 3.")
        

def find_highest_integral_frames_2d(
    array: np.ndarray,
    len_default: int = 40
) -> np.ndarray:
    if array.shape[0] <= len_default:
        return array
    else:
        array_copy = array.copy()
        means = array_copy.mean(axis=1)
        integrals = centered_moving_average(arr=means, window_size=4)

        max_sum = 0
        max_sum_start_index = 0

        window_sum = np.sum(integrals[:len_default])
        max_sum = window_sum

        for i in range(len(integrals) - len_default - 1):
            window_sum = window_sum - integrals[i] + integrals[i + len_default]
            if window_sum > max_sum:
                max_sum = window_sum
                max_sum_start_index = i + 1

        array = array[max_sum_start_index:max_sum_start_index + len_default]
        return array


def find_highest_integral_frames_3d(
    array: np.ndarray,
    len_default: int = 40
) -> np.ndarray:
    if array.shape[0] <= len_default:
        return array
    else:
        array_copy = array.copy()
        means = array_copy.mean(axis=(1, 2))
        integrals = centered_moving_average(arr=means, window_size=4)

        max_sum = 0
        max_sum_start_index = 0

        window_sum = np.sum(integrals[:len_default])
        max_sum = window_sum

        for i in range(len(integrals) - len_default - 1):
            window_sum = window_sum - integrals[i] + integrals[i + len_default]
            if window_sum > max_sum:
                max_sum = window_sum
                max_sum_start_index = i + 1

        array = array[max_sum_start_index:max_sum_start_index + len_default]
        return array