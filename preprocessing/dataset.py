'''
dataset.py

This file contains the Dataset class, which is used to create the dataset for the model.

The dataset is composed of the following data:
[LINK]

TODO:
    - add the link to the data
    - add the link to the paper
    - others...
'''

import os
import h5py
from torch.utils.data import Dataset

class Dataset(Dataset):
    '''
    Class to create the dataset with rdn or
    mDoppler data for the model
    '''
    
    def __init__(self,
                 TYPE:str,
                 dirname:str,
                 filename:str,
                 features_transform=None,
                 labels_transform=None):
        '''
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
        '''
        
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
        
    
    def __len__(self) -> int:
        '''
        Get the length of the dataset
        
        Returns
        -------
        int
            Length of the dataset
        '''
        return len(self.labels)
    
    
    def __getitem__(self,
                    idx:int)->tuple:
        '''
        Get the item at the given index
        
        Parameters
        ----------
        idx : int
            Index of the item to get
            
        Returns
        -------
        tuple
            Tuple containing the two rdn data and the label
        '''
        features_1 = self.features_1[idx]
        features_2 = self.features_2[idx]
        label = self.labels[idx]
        
        if self.features_transform:
            features_1 = self.features_transform(features_1)
            features_2 = self.features_transform(features_2)
            
        if self.labels_transform:
            label = self.labels_transform(label)
            
        return features_1, features_2, label