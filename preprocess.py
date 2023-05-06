'''
preprocess.py

Data preprocessing batch by batch

Usage:
    preprocess.py reading (--data_path <data_path>) (--output_path <output_path>) (--subjects <subjects>...) <sets> <do_rdn> <do_mDoppler> <verbose>

WRITE DOCSTRING


'''
import numpy as np
from docopt import docopt
from preprocessing.DataReader import DataReader
from preprocessing.DataCutter import DataCutter
from preprocessing.DataProcess import DataProcess

def reading(data_path:str,
            subjects:list,
            sets:list,
            do_rdn:bool,
            do_mDoppler:bool,
            verbose:int):
    '''
    Read the data and save it in a .npz file

    Parameters
    ----------
    data_path : str
        Path to the data
    output_path : str
        Path to save the data
    subjects : list
        List of subjects to read
    sets : list
        List of sets to read
    do_rdn : bool
        Whether to read the rdn data
    do_mDoppler : bool
        Whether to read the mDoppler data
    verbose : int
        Verbosity level
    '''
    if verbose > 0:
        print('Reading data...')
    dr = DataReader(subjects=subjects, sets=sets, do_rdn=do_rdn, do_mDoppler=do_mDoppler, data_dir=data_path)
    if verbose > 1:
        print("Removing static bins...")
    dr.remove_static_bins()
    if verbose > 1:
        print("Removing silence...")
    dr.crop()
    if verbose > 1:
        print("Rescaling...")
    dr.rescaling(method='norm')
    if do_rdn:
        if verbose > 1:
            print("Filtering...")
        dr.filter_mDoppler(size=(21, 11), sigma=15)
    return dr
            

def cutting(data:DataReader,
            output_path:str,
            subjects:list,
            sets:list,
            verbose:int):
    '''
    Cut the data and save it in a .npz file

    Parameters
    ----------
    data_path : str
        Path to the data
    output_path : str
        Path to save the data
    batch_size : int
        Batch size
    n_samples : int
        Number of samples to extract from each batch
    case : int
        Case to preprocess
    verbose : int
        Verbosity level
    '''
    if verbose > 0:
        print('Cutting data...')
    dc = DataCutter(data=data)
    if verbose > 1:
        print("Cutting...")
    dc.cut()
    if verbose > 1:
        print("Creating labels...")
    dc.create_labels_list()
    if verbose > 1:
        print("Converting labels to int...")
    dc.labels_to_int()
    if verbose > 1:
        print("Saving cutted data...")
    if len(subjects) > 1:
        dc.save(path=output_path,
            filename=f'cutted_data_{subjects[0]}_{subjects[-1]}_{sets[0]}_{sets[-1]}.npz')
    else:
        dc.save(path=output_path,
                filename=f'cutted_data_{subjects[0]}_{sets[0]}_{sets[-1]}.npz')
    return dc
    
def process(data:DataCutter,
            output_path:str,
            subjects:list,
            sets:list,
            verbose:int):
    '''
    Process the data and save it in a .npz file

    Parameters
    ----------
    data_path : str
        Path to the data
    output_path : str
        Path to save the data
    batch_size : int
        Batch size
    n_samples : int
        Number of samples to extract from each batch
    case : int
        Case to preprocess
    verbose : int
        Verbosity level
    '''
    if verbose > 0:
        print('Processing data...')
    dp = DataProcess(data=data)
    if verbose > 1:
        print("Separating actions in time...")
    dp.cut_time(loc='random')
    if verbose > 1:
        print("Padding...")
    dp.padding(padding=40)
    if verbose > 1:
        print("Saving processed data...")
    if len(subjects) > 1:
        dp.save(path=output_path,
                filename=f'processed_data_{subjects[0]}_{subjects[-1]}_{sets[0]}_{sets[-1]}.npz')
    else:
        dp.save(path=output_path,
                filename=f'processed_data_{subjects[0]}_{sets[0]}_{sets[-1]}.npz')
    
    return dp

def preprocess(data_path:str,
               output_path:str,
               subjects:list,
               sets:list,
               do_rdn:bool,
               do_mDoppler:bool,
               verbose:int):
    '''
    Preprocess the data and save it in a .npz file

    Parameters
    ----------
    data_path : str
        Path to the data
    output_path : str
        Path to save the data
    batch_size : int
        Batch size
    n_samples : int
        Number of samples to extract from each batch
    case : int
        Case to preprocess
    verbose : int
        Verbosity level
    '''
    dr = reading(data_path=data_path,
                 subjects=subjects,
                 sets=sets,
                 do_rdn=do_rdn,
                 do_mDoppler=do_mDoppler,
                 verbose=verbose)

    print('Cutting data...')
    dc = cutting(data=dr,
                 output_path=output_path,
                 subjects=subjects,
                 sets=sets,
                 verbose=verbose)

    print('Processing data...')
    dp = process(data=dc,
                 output_path=output_path,
                 subjects=subjects,
                 sets=sets,
                 verbose=verbose)
    
if __name__ == '__main__':
    args = docopt(__doc__)
    
    pass