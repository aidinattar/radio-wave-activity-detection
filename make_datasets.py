"""
make_datasets.py

Generate training and test datasets from the original H5 file.

Usage:
    make_datasets.py [--augment]
    
Options:
    -h --help       Show this screen.
    -a --augment    Augment the data by applying data augmentation techniques to the samples. [default: False]
"""


import os
import h5py
import numpy as np
from utils import augmentation
from sklearn.utils import class_weight
from docopt import docopt

def split_train_test(
    input_file: str,
    train_file: str,
    test_file: str,
    split_ratio:float=0.8
):
    """
    Split an H5 file into training and test data.
    
    Parameters
    ----------
    input_file : str
        Path to the input H5 file
    train_file : str
        Path to the output training H5 file
    test_file : str
        Path to the output test H5 file
    split_ratio : float, optional
        Ratio of training data to test data. The default is 0.8.
    """
    # Open the original H5 file
    with h5py.File(input_file, 'r') as f:
        # Create new H5 files for training and test data
        with h5py.File(train_file, 'w') as train_f, h5py.File(test_file, 'w') as test_f:
            # Iterate over the datasets in the original H5 file
            for dataset_name, dataset in f.items():
                # Get the dataset shape
                dataset_shape = dataset.shape
                train_shape = (int(dataset_shape[0] * split_ratio), *dataset_shape[1:])
                test_shape = (dataset_shape[0] - train_shape[0], *dataset_shape[1:])
                # Create datasets in the new H5 files with the same shape as the original dataset
                train_dataset = train_f.create_dataset(dataset_name, shape=train_shape, dtype=dataset.dtype,
                                                       chunks=True, compression="gzip")
                test_dataset = test_f.create_dataset(dataset_name, shape=test_shape, dtype=dataset.dtype,
                                                      chunks=True, compression="gzip")
                
                train_index, test_index = 0, 0
                # Copy the data chunk by chunk
                for i in range(0, dataset_shape[0], 1000):
                    chunk = dataset[i:i+1000]
                    
                    # Calculate the split index based on the split ratio
                    split_index = int(chunk.shape[0] * split_ratio)
                    
                    # Split the chunk into training and test data
                    train_chunk = chunk[:split_index]
                    test_chunk = chunk[split_index:]
                    
                    # Write the data to the new H5 files
                    train_dataset[train_index:train_index+train_chunk.shape[0]] = train_chunk
                    test_dataset[test_index:test_index+test_chunk.shape[0]] = test_chunk

                    # Update the current positions in each dataset
                    train_index += train_chunk.shape[0]
                    test_index += test_chunk.shape[0]


def augment_data(
    data: np.ndarray,
    labels: np.ndarray,
    augmentation_factors: dict
):
    """
    Augment the data by applying data augmentation techniques to the samples.
    
    Parameters
    ----------
    data : np.ndarray
        Array of samples
    labels : np.ndarray
        Array of labels
    augmentation_factors : dict
        Dictionary of augmentation factors for each class
        
    Returns
    -------
    augmented_data : np.ndarray
        Array of augmented samples
    augmented_labels : np.ndarray
        Array of augmented labels
    """
    augmented_data = []
    augmented_labels = []

    for i in range(len(data)):
        sample = data[i]
        label = labels[i]

        for _ in range(augmentation_factors[label]):
            # Apply data augmentation techniques to the sample
            augmented_sample_1 = augmentation.time_mask(sample)
            augmented_sample_2 = augmentation.doppler_mask(sample)
            augmented_sample_3 = augmentation.time_doppler_mask(sample)

            # Add augmented samples and labels to the list
            augmented_data.extend([augmented_sample_1, augmented_sample_2, augmented_sample_3])
            augmented_labels.extend([label, label, label])

    augmented_data = np.array(augmented_data)
    augmented_labels = np.array(augmented_labels)

    return augmented_data, augmented_labels


def add_augmented_samples_to_h5(
    input_file: str,
    output_file: str,
    augmentation_factors: dict,
    batch_size:int=1000
):
    """
    Add augmented samples to an H5 file.
    
    Parameters
    ----------
    input_file : str
        Path to the input H5 file
    output_file : str
        Path to the output H5 file
    augmentation_factors : dict
        Dictionary of augmentation factors for each class
    batch_size : int, optional
        Batch size for loading data from the H5 file. The default is 1000.
        
    Returns
    -------
    None
    """
    label_done = False
    # Load the original H5 file
    with h5py.File(input_file, 'r') as f:
        input_h5 = h5py.File(input_file, 'r')

        # Open the output file in 'a' mode to append to existing datasets
        output_h5 = h5py.File(output_file, 'a')

        for dataset_name in input_h5.keys():
            # create a dataset in the output H5 file with the same
            # shape as the original dataset
            dataset_shape = input_h5[dataset_name].shape
            dataset_dtype = input_h5[dataset_name].dtype
            output_h5.create_dataset(
                dataset_name,
                shape=dataset_shape,
                dtype=dataset_dtype,
                chunks=True,
                compression="gzip",
                maxshape=(None, *dataset_shape[1:])
            )
            
            # copy the data chunk by chunk
            for i in range(0, dataset_shape[0], batch_size):
                chunk = input_h5[dataset_name][i:i+batch_size]
                
                output_h5[dataset_name][i:i+chunk.shape[0]] = chunk
            
            
            if dataset_name == 'labels':
                # Skip the 'labels' dataset as it will be processed separately
                continue

            data_group = input_h5[dataset_name]
            labels_group = input_h5['labels']

            num_samples = data_group.shape[0]
            num_batches = (num_samples + batch_size - 1) // batch_size

            for batch_index in range(num_batches):
                start_index = batch_index * batch_size
                end_index = min((batch_index + 1) * batch_size, num_samples)

                # Load a batch of data and labels
                data = data_group[start_index:end_index]
                labels = labels_group[start_index:end_index]

                # Augment the data
                augmented_data, augmented_labels = augment_data(data, labels, augmentation_factors)

                # Append augmented samples to the original data in the output H5 file
                output_h5[dataset_name].resize((output_h5[dataset_name].shape[0] + augmented_data.shape[0]), axis=0)
                output_h5[dataset_name][-augmented_data.shape[0]:] = augmented_data

                if not label_done:
                    # Append labels to the output H5 file
                    output_h5['labels'].resize((output_h5['labels'].shape[0] + augmented_labels.shape[0]), axis=0)
                    output_h5['labels'][-augmented_labels.shape[0]:] = augmented_labels

            label_done = True
            
    # Close the input and output H5 files
    input_h5.close()
    output_h5.close()

    print("Augmented samples added to the H5 file.")


if __name__=='__main__':
    args = docopt(__doc__)
    
    TYPE = 'mDoppler'
    channels = 2

    dirname = 'DATA_preprocessed'
    filename = 'processed_data_mDoppler.h5'

    file = h5py.File(
        os.path.join(dirname, filename),
        'r'
    )

    filename = os.path.join(dirname, filename)
    train_filename = os.path.join(dirname, f'train_{TYPE}_{channels}channels.h5')
    test_filename = os.path.join(dirname, f'test_{TYPE}_{channels}channels.h5')

    # remove the destination file if it already exists
    if os.path.exists(train_filename):
        os.remove(train_filename)
        
    if os.path.exists(test_filename):
        os.remove(test_filename)
        
    split_train_test(
        filename,
        train_filename,
        test_filename,
        split_ratio=0.8
    )
    
    if args['--augment']:
        train_augmented_filename = os.path.join(dirname, f'train_{TYPE}_{channels}channels_augmented.h5')

        labels = h5py.File(train_filename, 'r')['labels'][:]

        augmentation_factor = 3

        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )

        class_weights = np.ceil(class_weights/class_weights.max() * augmentation_factor).astype(int)

        source_file = train_filename
        destination_file = train_augmented_filename

        # remove the destination file if it already exists
        if os.path.exists(destination_file):
            os.remove(destination_file)
        #shutil.copy(source_file, destination_file)

        print("HDF5 file copied from '{}' to '{}'.".format(source_file, destination_file))

        class_weights_dict = dict(zip(np.unique(labels), class_weights))

        add_augmented_samples_to_h5(
            train_filename,
            train_augmented_filename,
            augmentation_factors=class_weights_dict
        )