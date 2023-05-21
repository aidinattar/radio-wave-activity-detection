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
import numpy as np

from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, path:str, file:str, transform=None, type:str='mDoppler'):
        """
        Constructor

        Parameters
        ----------
        path : str
            Path to the dataset
        file : str
            Name of the file containing the dataset
        transform : callable, optional
            Transform to apply to the data. The default is None.
        type : str, optional
            Type of data to load. The default is 'mdoppler'.
            Possible values are:
                'mdoppler': mDoppler data
                'rdn': rdn data
        """
        # Load the data
        data = np.load(os.path.join(path, file), allow_pickle=True)

        if type == 'mDoppler':
            self.x = np.concatenate((data['mDoppler_1'], data['mDoppler_2']), axis=0)
        elif type == 'rdn':
            self.x = np.concatenate((data['rdn_1'], data['rdn_2']), axis=0)
        else:
            raise ValueError('Invalid type of data')

        self.y = np.concatenate((data['labels'], data['labels']), axis=0)
        self.labels_dict = data['labels_dict']
        self.transform = transform
        self.type = type


    @classmethod
    def from_data(cls, x:np.ndarray, y:np.ndarray, labels_dict:dict, transform=None, type:str='mDoppler')->Dataset:
        """
        Constructor from data

        Parameters
        ----------
        x : np.ndarray
            Data
        y : np.ndarray
            Labels
        labels_dict : dict
            Dictionary containing the labels
        transform : callable, optional
            Transform to apply to the data. The default is None.
        type : str, optional
            Type of data to load. The default is 'mdoppler'.
            Possible values are:
                'mdoppler': mDoppler data
                'rdn': rdn data

        Returns
        -------
        dataset : Dataset
            Dataset created from the data
        """
        dataset = cls.__new__(cls)
        dataset.x = x
        dataset.y = y
        dataset.labels_dict = labels_dict
        dataset.transform = transform
        dataset.type = type
        return dataset

    def __len__(self)->int:
        """
        Returns the length of the dataset
        
        Returns
        -------
        length : int
            Length of the dataset
        """
        return len(self.x)

    def __getitem__(self, idx)->tuple:
        """
        Returns the data at the given index

        Parameters
        ----------
        idx : int
            Index of the data to return
            
        Returns
        -------
        x : np.ndarray
            Data
        y : np.ndarray
            Label
        """
        x = self.x[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def concat(self, other: Dataset) -> Dataset:
        """
        Concatenate the dataset with another dataset

        Parameters
        ----------
        other : Dataset
            Dataset to concatenate with

        Returns
        -------
        dataset : Dataset
            Concatenated dataset
        """
        x = np.concatenate((self.x, other.x), axis=0)
        y = np.concatenate((self.y, other.y), axis=0)
        return Dataset.from_data(x, y, self.labels_dict, self.transform, self.type)
    
    
    def drop_duplicates(self):
        """
        Drop the duplicates from the dataset
        """
        # Get the unique values
        _, indices = np.unique(self.x, return_index=True, axis=0)
        
        # Drop the duplicates
        self.x = self.x[indices]
        self.y = self.y[indices]