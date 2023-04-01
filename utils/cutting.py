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

### Doubt: can we extract indexes according to a normal
### distribution without replacements?
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
        #### NO np.random.normal
        #### USE np.random.choice with custom probability
        #### Assign probabilities accoreding to a normal distribution
        normal_bins = np.random.normal(
            loc=mean,
            scale=std,
            size=len_default
        )
        normal_bins = np.round(normal_bins).astype(int)
        normal_bins.sort()
        return array[normal_bins]
