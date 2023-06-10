"""
dataset.py

This file contains the Dataset class, which is used to create the dataset for the model.

The dataset is composed of the following data:
[LINK]

TODO:
    - add the link to the data
    - add the link to the paper
    - others...
"""

import os
import h5py
import numpy as np
from torch import randperm
from torch.utils.data import Dataset

class Dataset2Channels(Dataset):
    """
    Class to create the dataset with rdn or
    mDoppler data for the model considering 
    two input channels
    """
    
    def __init__(self,
                 TYPE:str,
                 dirname:str,
                 filename:str,
                 features_transform=None,
                 labels_transform=None,
                 combine_channels=True):
        """
        Initialize the dataset
        
        Parameters
        ----------
        TYPE : str
            Type of data to load.
            Possible values are:
                'mDoppler': mDoppler data
                'rdn': rdn data
        dirname : str
            Name of the directory containing the dataset in .h5 format
        filename : str
            Name of the file containing the dataset in .h5 format
        transform : callable, optional
            Transform to apply to the data. The default is None.
        combine_channels : bool, optional
            If True, the two channels are combined.
        """
        
        self.features_1 = h5py.File(
            name=os.path.join(
                dirname,
                filename),
            mode='r'
        )[f'{TYPE}_1']
        
        self.features_2 = h5py.File(
            name=os.path.join(
                dirname,
                filename),
            mode='r'
        )[f'{TYPE}_2']
        
        self.labels = h5py.File(
            name=os.path.join(
                dirname,
                filename),
            mode='r'
        )['labels']
        
        self.features_transform = features_transform
        self.labels_transform = labels_transform
        self.combine_channels = combine_channels
        self.TYPE = TYPE
        self.indices = np.arange(len(self))
        
    
    def __len__(self) -> int:
        """
        Get the length of the dataset
        
        Returns
        -------
        int
            Length of the dataset
        """
        return len(self.labels)
    
    
    def __getitem__(self,
                    idx:int)->tuple:
        """
        Get the item at the given index
        
        Parameters
        ----------
        idx : int
            Index of the item to get
            
        Returns
        -------
        tuple
            Tuple containing the two rdn data and the label
        """
        idx = self.indices[idx]
        features_1 = self.features_1[idx]
        features_2 = self.features_2[idx]
        label = self.labels[idx]

        if self.labels_transform:
            label = self.labels_transform(label)
            
        if self.combine_channels:
            features_1 = np.expand_dims(features_1, axis=0)
            features_2 = np.expand_dims(features_2, axis=0)
            features = np.concatenate((features_1, features_2), axis=0)
            
            if self.features_transform:
                if self.TYPE == 'mDoppler':
                    features = self.features_transform(features).permute(1, 0, 2)
                else:
                    features = self.features_transform(features)
            
            return features, label

        if self.features_transform:
            features_1 = self.features_transform(features_1)
            features_2 = self.features_transform(features_2)

        return features_1, features_2, label

    
    def shuffle(self):
        # Shuffle the indices
        shuffled_indices = randperm(len(self))
        
        # Update the indices for subsequent data retrieval
        self.indices = shuffled_indices


    def __iter__(self):
        # Shuffle the dataset when creating an iterator
        self.shuffle()
        
        # Return an iterator over the shuffled indices
        return iter(self.indices.tolist())
    
    
    
class Dataset1Channel(Dataset):
    """
    Class to create the dataset with rdn or
    mDoppler data for the model with one input channel
    """
    
    def __init__(self,
                 TYPE:str,
                 dirname:str,
                 filename:str,
                 features_transform=None,
                 labels_transform=None,
                 channel:int=1):
        """
        Initialize the dataset
        
        Parameters
        ----------
        TYPE : str
            Type of data to load.
            Possible values are:
                'mDoppler': mDoppler data
                'rdn': rdn data
        dirname : str
            Name of the directory containing the dataset in .h5 format
        filename : str
            Name of the file containing the dataset in .h5 format
        transform : callable, optional
            Transform to apply to the data. The default is None.
        channel : int, optional
            Channel to load. The default is 1.
            
        Raises
        ------
        AssertionError
            If the channel is not 1 or 2
        """
        
        assert channel in [1, 2], "Channel must be 1 or 2"
        
        self.features = h5py.File(
            name=os.path.join(
                dirname,
                filename),
            mode='r'
        )[f'{TYPE}_{channel}']
        
        self.labels = h5py.File(
            name=os.path.join(
                dirname,
                filename),
            mode='r'
        )['labels']
        
        self.features_transform = features_transform
        self.labels_transform = labels_transform
        self.TYPE = TYPE
        self.indices = np.arange(len(self))
        
    
    def __len__(self) -> int:
        """
        Get the length of the dataset
        
        Returns
        -------
        int
            Length of the dataset
        """
        return len(self.labels)
    
    
    def __getitem__(self,
                    idx:int)->tuple:
        """
        Get the item at the given index
        
        Parameters
        ----------
        idx : int
            Index of the item to get
            
        Returns
        -------
        tuple
            Tuple containing the two rdn data and the label
        """
        idx = self.indices[idx]
        features = self.features[idx]
        label = self.labels[idx]
        
        if self.features_transform:
            features = self.features_transform(features)
            
        if self.labels_transform:
            label = self.labels_transform(label)
            
        return features, label
    
    
    def shuffle(self):
        # Shuffle the indices
        shuffled_indices = randperm(len(self))
        
        # Update the indices for subsequent data retrieval
        self.indices = shuffled_indices

        
    def __iter__(self):
        # Shuffle the dataset when creating an iterator
        self.shuffle()
        
        # Return an iterator over the shuffled indices
        return iter(self.indices.tolist())