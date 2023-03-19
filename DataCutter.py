'''
DataCutter.py

This file contains the DataCutter class, which is used to cut the data into the signals
according to the timestamps provided by the timestamp_speech system.
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DataReader import DataReader

class DataCutter(object):
    '''
    Class to cut the data into the signals
    '''

    def __init__(self, data: DataReader):
        '''
        Constructor
        '''
        self.data = data
        self.timestamps = self.data.timestamps

    