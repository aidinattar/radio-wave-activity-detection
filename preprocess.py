"""
preprocess.py

Data preprocessing batch by batch

Usage:
    preprocess.py (--data_path <data_path>) (--output_path <output_path>) (--all | --selection <start-stop>...) [--do_rdn] [--do_mDoppler] (--verbose=<verbose>) [--tqdm]

Options:
    -h --help                       Show this screen.
    --data_path <data_path>         Path to the data
    --output_path <output_path>     Path to save the data
    --all                              Read all the data
    --selection <start-stop>        Read a selection of the data
    --do_rdn                        Whether to read the rdn data
    --do_mDoppler                   Whether to read the mDoppler data
    --tqdm                          Whether to use tqdm
    --verbose=<verbose>             Verbosity level
    
We suggest to avoid using tqdm together with the verbose option,
as it may cause some problems with the progress bar.

Example:
    python preprocess.py --data_path=DATA --output_path=DATA_preprocessed --all --do_mDoppler --verbose=10
"""
import os
import h5py
import numpy as np
from tqdm import tqdm
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
    """
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
        
    Returns
    -------
    dr : DataReader
        DataReader object
    """
    if verbose > 0:
        print('Reading data...')
    dr = DataReader(subjects=subjects, sets=sets, do_rdn=do_rdn, do_mDoppler=do_mDoppler, data_dir=data_path)
    if verbose > 1:
        print("Removing static bins...")
    dr.remove_static_bins()
    if verbose > 1:
        print("Removing silence...")
    dr.crop(start=22, stop=102)
    if verbose > 1:
        print("Rescaling...")
    dr.rescaling(method='norm')
    #if verbose > 1:
    #    print("Filtering...")
    #if do_mDoppler:
    #    if verbose > 1:
    #        print("Filtering...")
    #    dr.filter_mDoppler(size=(21, 11), sigma=15)
    return dr
            

def cutting(data:DataReader,
            output_path:str,
            subjects:list,
            sets:list,
            verbose:int):
    """
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
        
    Returns
    -------
    dc : DataCutter
        DataCutter object
    """

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
    #if len(subjects) > 1:
    #    dc.save(path=output_path,
    #        filename=f'cutted_data_{subjects[0]}_{subjects[-1]}_{sets[0]}_{sets[-1]}.npz')
    #else:
    #    dc.save(path=output_path,
    #            filename=f'cutted_data_{subjects[0]}_{sets[0]}_{sets[-1]}.npz')
    
    return dc
    
def process(data:DataCutter,
            output_path:str,
            subjects:list,
            sets:list,
            verbose:int):
    """
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
        
    Returns
    -------
    dp : DataProcess
        DataProcess object
    """
    if verbose > 0:
        print('Processing data...')
    dp = DataProcess(data=data)
    dp.remove_long_actions(
        threshold=100
    )
    if verbose > 1:
        print("Separating actions in time...")
    dp.cut_time(
        len_default=40,
        loc='max-integral',
        #threshold=1
    )
    if verbose > 1:
        print("Padding...")
    dp.padding(
        padding=40,
        mode='last-frame')
    if verbose > 1:
        print("Saving processed data...")
    #if len(subjects) > 1:
    #    dp.save(path=output_path,
    #            filename=f'processed_data_{subjects[0]}_{subjects[-1]}_{sets[0]}_{sets[-1]}.npz')
    #else:
    #    dp.save(path=output_path,
    #            filename=f'processed_data_{subjects[0]}_{sets[0]}_{sets[-1]}.npz')
    
    if dp.is_empty():
        return False
    
    return dp


def save_h5(data:DataProcess,
            output_path:str,
            iteration:int,
            do_rdn:bool,
            do_mDoppler:bool,
            verbose:int):
    """
    Save the data in a .h5 file
    
    Parameters
    ----------
    data : DataProcess
        Data to save
    output_path : str
        Path to save the data
    iteration : int
        Iteration number
    do_rdn : bool
        Whether to save the rdn data
    do_mDoppler : bool
        Whether to save the mDoppler data
    verbose : int
        Verbosity level
        
    Returns
    -------
    file : h5py.File
        HDF5 file with the data
    """
    
    if verbose > 0:
        print('Saving data...')
        
    if iteration == 0:
        type = 'rdn' if do_rdn else 'mDoppler' if do_mDoppler else 'rdn_mDoppler'
        filename = f'processed_data_{type}.h5'
        
        # Create or overwrite the HDF5 file
        file = h5py.File(
            name=os.path.join(
                output_path,
                filename
            ),
            mode='w'
        )
        
        if do_rdn:
            # Get the shape of an inner list
            inner_shape = np.shape(data.rdn_1[0])
            # Convert to a numpy array
            rdn_1 = np.array(data.rdn_1).\
                reshape((len(data.rdn_1), *inner_shape))
            file.create_dataset(
                name='rdn_1',
                data=rdn_1,
                dtype='float32',
                compression='gzip',
                maxshape=(None, *inner_shape)
            )

            # Get the shape of an inner list
            inner_shape = np.shape(data.rdn_2[0])
            # Convert to a numpy array
            rdn_2 = np.array(data.rdn_2).\
                reshape((len(data.rdn_2), *inner_shape))
            file.create_dataset(
                name='rdn_2',
                data=rdn_2,
                dtype='float32',
                compression='gzip',
                maxshape=(None, *inner_shape)
            )
        
        if do_mDoppler:
            # Get the shape of an inner list
            inner_shape = np.shape(data.mDoppler_1[0])
            # Convert to a numpy array
            mDoppler_1 = np.array(data.mDoppler_1).\
                reshape((len(data.mDoppler_1), *inner_shape))
            file.create_dataset(
                name='mDoppler_1',
                data=mDoppler_1,
                dtype='float32',
                compression='gzip',
                maxshape=(None, *inner_shape)
            )
            
            # Get the shape of an inner list
            inner_shape = np.shape(data.mDoppler_2[0])
            # Convert to a numpy array
            mDoppler_2 = np.array(data.mDoppler_2).\
                reshape((len(data.mDoppler_2), *inner_shape))
            file.create_dataset(
                name='mDoppler_2',
                data=mDoppler_2,
                dtype='float32',
                compression='gzip',
                maxshape=(None, *inner_shape)
            )
        
        labels = np.array(data.labels)
        file.create_dataset(
            name='labels',
            data=labels,
            dtype='int32',
            compression='gzip',
            maxshape=(None,)
        )
        
        #group = file.create_group(
        #    name='labels_dict'
        #)
        #for key, value in data.labels_dict.items():
        #    group.create_dataset(
        #        name=key,
        #        data=value,
        #        dtype='int32',
        #        compression='gzip'
        #    )
    else:
        type = 'rdn' if do_rdn else 'mDoppler' if do_mDoppler else 'rdn_mDoppler'
        filename = f'processed_data_{type}.h5'
        
        # Open the HDF5 file in append mode
        file = h5py.File(
            name=os.path.join(
                output_path,
                filename
            ),
            mode='a'
        )
        
        if do_rdn:
            # Get the shape of an inner list
            inner_shape = np.shape(data.rdn_1[0])
            # Convert to a numpy array
            rdn_1 = np.array(data.rdn_1).\
                reshape((len(data.rdn_1), *inner_shape))
            # Get the shape of an inner list
            inner_shape = np.shape(data.rdn_2[0])
            # Convert to a numpy array
            rdn_2 = np.array(data.rdn_2).\
                reshape((len(data.rdn_2), *inner_shape))
                
            file['rdn_1'].resize((file['rdn_1'].shape[0] + rdn_1.shape[0]), axis=0)
            file['rdn_1'][-rdn_1.shape[0]:] = rdn_1
            file['rdn_2'].resize((file['rdn_2'].shape[0] + rdn_2.shape[0]), axis=0)
            file['rdn_2'][-rdn_2.shape[0]:] = rdn_2
        
        if do_mDoppler:
            # Get the shape of an inner list
            inner_shape = np.shape(data.mDoppler_1[0])
            # Convert to a numpy array
            mDoppler_1 = np.array(data.mDoppler_1).\
                reshape((len(data.mDoppler_1), *inner_shape))
            # Get the shape of an inner list
            inner_shape = np.shape(data.mDoppler_2[0])
            # Convert to a numpy array
            mDoppler_2 = np.array(data.mDoppler_2).\
                reshape((len(data.mDoppler_2), *inner_shape))
                
            file['mDoppler_1'].resize((file['mDoppler_1'].shape[0] + mDoppler_1.shape[0]), axis=0)
            file['mDoppler_1'][-mDoppler_1.shape[0]:] = mDoppler_1
            file['mDoppler_2'].resize((file['mDoppler_2'].shape[0] + mDoppler_2.shape[0]), axis=0)
            file['mDoppler_2'][-mDoppler_2.shape[0]:] = mDoppler_2
        
        #if data.labels_dict != file['labels_dict']:
        #    mapping = {data.labels_dict[key]: file['labels_dict'][key] for key in data.labels_dict.keys()}
        #    data.labels = [mapping[value] for value in data.labels]
        
        labels = np.array(data.labels)
        file['labels'].resize((file['labels'].shape[0] + labels.shape[0]), axis=0)
        file['labels'][-labels.shape[0]:] = labels

    return file

def preprocess(data_path:str,
               output_path:str,
               subjects:list,
               sets:list,
               do_rdn:bool,
               do_mDoppler:bool,
               verbose:int):
    """
    Preprocess the data and save it in a .npz file

    Parameters
    ----------
    data_path : str
        Path to the data
    output_path : str
        Path to save the data
    subjects : list of str
        List of subjects to read
    sets : list of str
        List of sets to read
    do_rdn : bool
        Whether to read the rdn data
    do_mDoppler : bool
        Whether to read the mDoppler data
    verbose : int
        Verbosity level
        
    Returns
    -------
    dp : DataProcess
        Processed data
    """
    dr = reading(data_path=data_path,
                 subjects=subjects,
                 sets=sets,
                 do_rdn=do_rdn,
                 do_mDoppler=do_mDoppler,
                 verbose=verbose)

    dc = cutting(data=dr,
                 output_path=output_path,
                 subjects=subjects,
                 sets=sets,
                 verbose=verbose)

    dp = process(data=dc,
                 output_path=output_path,
                 subjects=subjects,
                 sets=sets,
                 verbose=verbose)
    
    return dp


if __name__ == '__main__':
    args = docopt(__doc__)
    
    data_path = args['--data_path']
    output_path = args['--output_path']
    if args['--all']:
        subjects = [f'subject_{i:02d}' for i in range(0, 24)]
        sets     = [f'set{i:03d}'      for i in range(0, 4)]
    
    elif args['--selection'] is not None:
        subjects = []

        # Iterate over each element of the selection list
        for sel in args['--selection']:
            # If the element contains a single index, treat it as the start and stop indices
            if '-' not in sel:
                start, stop = int(sel), int(sel)
            # If the element contains a range, split it into the start and stop indices
            else:
                start, stop = map(int, sel.split('-'))

            # Create the subjects list for the current selection range
            subjects_range = [f'subject_{i:02d}' for i in range(start, stop+1)]

            # Add the subjects for the current selection range to the overall subjects list
            subjects.extend(subjects_range)

    else:
        raise ValueError('No option selected')
        
    sets = [f'set{i:03d}' for i in range(0, 4)]

    do_rdn = args['--do_rdn']
    do_mDoppler = args['--do_mDoppler']
    verbose = int(args['--verbose'])
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    progress = tqdm(enumerate(subjects)) \
                if args['--tqdm'] \
                else enumerate(subjects)
    
    for i, subject in progress:
        
        if verbose > 0:
            print(f'Processing subject {subject}')

        dp = preprocess(
            data_path=data_path,
            output_path=output_path,
            subjects=[subject],
            sets=sets,
            do_mDoppler=do_mDoppler,
            do_rdn=do_rdn,
            verbose=verbose)

        if dp:
            file = save_h5(
                data=dp,
                output_path=output_path,
                iteration=i,
                do_mDoppler=do_mDoppler,
                do_rdn=do_rdn,
                verbose=verbose
            )
    
    if "file" in locals():
        file.close()