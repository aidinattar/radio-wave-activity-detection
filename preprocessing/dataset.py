'''
dataset.py

This file contains the Dataset class, which is used to create the dataset for the model.

The dataset is composed of the following data:
[LINK]

TODO:
    - add the link to the data
    - add the link to the paper
'''

import os
import numpy as np

from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init_(self, path:str, file:str, transform=None, type:str='mdoppler'):
        '''
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
        '''
        # Load the data
        data = np.load(os.path.join(path, file))

        if type == 'mdoppler':
            self.x = np.concatenate((data['mdoppler_1'], data['mdoppler_2']), axis=0)
        elif type == 'rdn':
            self.x = np.concatenate((data['rdn_1'], data['rdn_2']), axis=0)
        else:
            raise ValueError('Invalid type of data')

        self.y = np.concatenate(data['labels'], data['labels'], axis=0)
        self.labels_dict = data['labels_dict']
        self.transform = transform

    def __len__(self):
        '''
        Returns the length of the dataset
        '''
        return len(self.x)

    def __getitem__(self, idx):
        '''
        Returns the data at the given index

        Parameters
        ----------
        idx : int
            Index of the data to return
        '''
        x = self.x[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y