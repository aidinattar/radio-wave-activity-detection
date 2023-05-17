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

class rdnDataset(Dataset):
    '''
    Class to create the dataset with rdn data for the model
    '''
    
    def __init__(self,
                 dirname:str,
                 filename:str,
                 transform=None,):
        '''
        Initialize the dataset
        
        Parameters
        ----------
        dirname : str
            Name of the directory containing the dataset in .h5 format
        filename : str
            Name of the file containing the dataset in .h5 format
        transform : callable, optional
            Transform to apply to the data. The default is None.
        '''
        
        self.rdn_1 = h5py.File(
            name=os.path.join(
                dirname,
                filename),
            mode='r'
        )['rdn_1']
        
        self.rdn_2 = h5py.File(
            name=os.path.join(
                dirname,
                filename),
            mode='r'
        )['rdn_2']
        
        self.labels = h5py.File(
            name=os.path.join(
                dirname,
                filename),
            mode='r'
        )['labels']
        
        self.labels_dict = h5py.File(
            name=os.path.join(
                dirname,
                filename),
            mode='r'
        )['labels_dict']
        
        self.transform = transform
        
    
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
        rdn_1 = self.rdn_1[idx]
        rdn_2 = self.rdn_2[idx]
        label = self.labels[idx]
        
        if self.transform:
            rdn_1 = self.transform(rdn_1)
            rdn_2 = self.transform(rdn_2)
            
        return rdn_1, rdn_2, label