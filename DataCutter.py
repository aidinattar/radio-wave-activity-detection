'''
DataCutter.py

This file contains the DataCutter class, which is used to cut the data into the signals
according to the timestamps provided by the timestamp_speech system.
'''

import os
import numpy as np

from DataReader import DataReader

class DataCutter(object):
    '''
    Class to cut the data into the signals
    '''

    cut_done = False
    labels_extraction_done = False
    labels_to_int_done = False

    def __init__(self, data: DataReader):
        '''
        Constructor
        '''
        self.data = data
        self.timestamps = self.data.timestamp_speech

    def cut(self, conversion_factor:float=1):
        '''
        Cut the data into the signals according to the timestamps provided by the timestamp_speech system.
        '''
        self.conversion_factor = conversion_factor

        # Convert the timestamps to bins
        if not self.data.timestamp_to_bins_done:
            self.data.timestamp_to_bins(conversion_factor=self.conversion_factor)

        # Cut the data
        if self.data.do_mDoppler:
            self.cut_mDoppler()
        if self.data.do_rdn:
            self.cut_rdn()

        self.cut_done = True

    def cut_mDoppler(self):
        '''
        Cut the mDoppler data into the signals according to the timestamps provided by the timestamp_speech system.
        '''

        # Create the list of signals
        self.signals_mDoppler_1, self.signals_mDoppler_2 = ([] for i in range(2))

        # Divide data in two radars
        if not self.data.radar_division_done:
            self.data.radar_division()

        # Cut the data
        for i, (mDoppler_1, mDoppler_2) in enumerate(zip(self.data.mDoppler_1, self.data.mDoppler_2)):
            # Get the time bins
            time_bins = self.timestamps[i]['time_passed_bins'].values
            # Loop over the time bins
            for j in range(len(time_bins)):
                # Cut the data
                self.signals_mDoppler_1.append(mDoppler_1[time_bins[j]:time_bins[j+1] if j+1<len(time_bins) else None])
                self.signals_mDoppler_2.append(mDoppler_2[time_bins[j]:time_bins[j+1] if j+1<len(time_bins) else None])

    def cut_rdn(self):
        '''
        Cut the rdn data into the signals according to the timestamps provided by the timestamp_speech system.
        '''

        # Create the list of signals
        self.signals_rdn_1, self.signals_rdn_2 = ([] for i in range(2))

        # Cut the data
        for i, (rdn_1, rdn_2) in enumerate(zip(self.data.rdn_1, self.data.rdn_2)):
            # Get the time bins
            time_bins = self.timestamps[i]['time_passed_bins'].values
            # Loop over the time bins
            for j in range(len(time_bins)):
                # Cut the data
                self.signals_rdn_1.append(rdn_1[time_bins[j]:time_bins[j+1] if j+1<len(time_bins) else None])
                self.signals_rdn_2.append(rdn_2[time_bins[j]:time_bins[j+1] if j+1<len(time_bins) else None])


    def create_labels_list(self):
        '''
        Create a list of labels.
        '''
        # Create the list of labels
        self.labels = []

        for rilev in self.timestamps:
            self.labels += rilev['command'].values.tolist()

        self.labels_extraction_done = True

    def labels_to_int(self):
        '''
        Convert the labels from string to integers, associating each label to a number.
        '''

        # Create the list of labels
        if not self.labels_extraction_done:
            self.create_labels_list()

        # Find unique labels
        unique_labels = np.unique(self.labels)

        # Create the dictionary
        self.labels_dict = {label: i for i, label in enumerate(unique_labels)}

        # Convert the labels to integers
        self.labels = [self.labels_dict[label] for label in self.labels]

        self.labels_to_int_done = True

    def save(self, path:str='DATA_preprocessed', filename:str='data_cutted.npz'):
        '''
        Save the data.
        '''

        # Create the path
        path = os.path.join(path, filename)

        # Save the data
        np.savez(path, signals_mDoppler_1=self.signals_mDoppler_1, signals_mDoppler_2=self.signals_mDoppler_2, signals_rdn_1=self.signals_rdn_1, signals_rdn_2=self.signals_rdn_2, labels=self.labels, labels_dict=self.labels_dict)


    def load(self, path:str='DATA_preprocessed', filename:str='data.npz'):
        '''
        Load the data.
        '''

        # Create the path
        path = os.path.join(path, filename)

        # Load the data
        data = np.load(path)

        # Get the data
        self.signals_mDoppler_1 = data['signals_mDoppler_1']
        self.signals_mDoppler_2 = data['signals_mDoppler_2']
        self.signals_rdn_1      = data['signals_rdn_1']
        self.signals_rdn_2      = data['signals_rdn_2']
        self.labels             = data['labels']
        self.labels_dict        = data['labels_dict'].item()