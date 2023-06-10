"""
DataCutter.py

This file contains the DataCutter class, which is used to cut the data into the signals
according to the timestamps provided by the timestamp_speech system.
"""

import os
import numpy as np

from preprocessing.DataReader import DataReader
from exceptions               import OptionIsFalseError
from utils.constants          import NON_AGGREGATED_LABELS_DICT


class DataCutter(object):
    """
    Class to cut the data into the signals
    """

    cut_done = False
    labels_extraction_done = False
    labels_to_int_done = False

    signals_mDoppler_1 = np.array([])
    signals_mDoppler_2 = np.array([])
    signals_rdn_1 = np.array([])
    signals_rdn_2 = np.array([])
    labels = np.array([])
    labels_dict = np.array([])


    def __init__(self,
                 data: DataReader):
        """
        Constructor
        
        Parameters
        ----------
        data : DataReader
            DataReader object containing the data
        """
        self.data = data
        self.timestamps = self.data.timestamp_speech

        
    @classmethod
    def empty(cls):
        """
        Empty constructor
        """
        return cls(DataReader.empty())

    
    @classmethod
    def from_file(cls,
                  path:str,
                  file:str):
        """
        Constructor from file
        
        Parameters
        ----------
        path : str
            Path to the file
        file : str
            Name of the file
        """
        cutter = cls(DataReader.empty())
        cutter.load(path, file)
        return cutter


    def cut(self):
        """
        Cut the data into the signals according to the
        timestamps provided by the timestamp_speech system.
        """

        # Convert the timestamps to bins
        if not self.data.timestamp_to_bins_done:
            self.data.timestamp_to_bins()

        # Divide data in two radars
        if not self.data.radar_division_done:
            self.data.radar_division()

        # Cut the data
        if self.data.do_mDoppler:
            self.cut_mDoppler()
        if self.data.do_rdn:
            self.cut_rdn()

        self.cut_done = True


    def cut_mDoppler(self):
        """
        Cut the mDoppler data into the signals according
        to the timestamps provided by the timestamp_speech system.
        """

        if not self.data.do_mDoppler:
            raise OptionIsFalseError('do_mDoppler')

        # Create the list of signals
        self.signals_mDoppler_1, self.signals_mDoppler_2 = ([] for i in range(2))

        # Cut the data
        for i, (mDoppler_1, mDoppler_2) in enumerate(zip(self.data.mDoppler_1, self.data.mDoppler_2)):
            # Get the time bins
            time_bins = self.timestamps[i].n_bins.cumsum().shift().fillna(0).astype(int).values

            # Loop over the time bins
            for j in range(len(time_bins)):
                # Cut the data
                self.signals_mDoppler_1.append(mDoppler_1[time_bins[j]:time_bins[j+1] if j+1<len(time_bins) else None])
                self.signals_mDoppler_2.append(mDoppler_2[time_bins[j]:time_bins[j+1] if j+1<len(time_bins) else None])


    def cut_rdn(self):
        """
        Cut the rdn data into the signals according to the
        timestamps provided by the timestamp_speech system.
        """

        if not self.data.do_rdn:
            raise OptionIsFalseError('do_rdn')

                # Create the list of signals
        self.signals_rdn_1, self.signals_rdn_2 = ([] for i in range(2))

        # Cut the data
        for i, (rdn_1, rdn_2) in enumerate(zip(self.data.rdn_1, self.data.rdn_2)):
            # Get the time bins
            time_bins = self.timestamps[i].n_bins.cumsum().shift().fillna(0).astype(int).values

            # Loop over the time bins
            for j in range(len(time_bins)):
                # Cut the data
                self.signals_rdn_1.append(rdn_1[time_bins[j]:time_bins[j+1] if j+1<len(time_bins) else None])
                self.signals_rdn_2.append(rdn_2[time_bins[j]:time_bins[j+1] if j+1<len(time_bins) else None])


    def create_labels_list(self):
        """
        Create a list of labels.
        """
        # Create the list of labels
        self.labels = []

        for rilev in self.timestamps:
            self.labels += rilev['command'].values.tolist()

        self.labels_extraction_done = True

    def labels_to_int(self):
        """
        Convert the labels from string to integers, associating each label to a number.
        """

        # Create the list of labels
        if not self.labels_extraction_done:
            self.create_labels_list()

        # Convert the labels to integers
        self.labels = [NON_AGGREGATED_LABELS_DICT[label] for label in self.labels]

        self.labels_to_int_done = True

    def save(self,
             path:str='DATA_preprocessed',
             filename:str='data_cutted.npz'):
        """
        Save the data.
        
        Parameters
        ----------
        path : str
            Path to the file
        filename : str
            Name of the file
        """
        # Create the path
        path = os.path.join(path, filename)

        # Save the data
        np.savez(
            path,
            signals_mDoppler_1=np.array(self.signals_mDoppler_1, dtype=object),
            signals_mDoppler_2=np.array(self.signals_mDoppler_2, dtype=object),
            signals_rdn_1=np.array(self.signals_rdn_1, dtype=object),
            signals_rdn_2=np.array(self.signals_rdn_2, dtype=object),
            labels=self.labels,
            #labels_dict=self.labels_dict
        )


    def load(self,
             path:str='DATA_preprocessed',
             filename:str='data.npz'):
        """
        Load the data from a npz file.
        
        Parameters
        ----------
        path : str
            Path to the file
        filename : str
            Name of the file
        """

        # Create the path
        path = os.path.join(path, filename)

        # Load the data
        data = np.load(path, allow_pickle=True)

        # Get the data
        self.signals_mDoppler_1 = data['signals_mDoppler_1']
        self.signals_mDoppler_2 = data['signals_mDoppler_2']
        self.signals_rdn_1      = data['signals_rdn_1']
        self.signals_rdn_2      = data['signals_rdn_2']
        self.labels             = data['labels']
        #self.labels_dict        = data['labels_dict'].item()

        if self.signals_mDoppler_1.size != 0:
            self.data.do_mDoppler = True
        if self.signals_rdn_1.size != 0:
            self.data.do_rdn = True