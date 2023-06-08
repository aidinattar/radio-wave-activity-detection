"""
DataProcess.py

This file contains the DataProcess class, which is used to process the data.
"""

import os
import numpy as np

from preprocessing.DataCutter import DataCutter
from exceptions               import OptionIsFalseError
from utils                    import cutting

class DataProcess(object):
    """
    Class to process the data
    """

    def __init__(self,
                 data: DataCutter):
        """
        Constructor
        
        Parameters
        ----------
        data : DataCutter
            DataCutter object containing the data
        """
        self.data = data
        self.do_rdn = self.data.data.do_rdn
        self.do_mDoppler = self.data.data.do_mDoppler
        self.labels = self.data.labels
        #self.labels_dict = self.data.labels_dict

        self.rdn_1 = self.data.signals_rdn_1
        self.rdn_2 = self.data.signals_rdn_2
        self.mDoppler_1 = self.data.signals_mDoppler_1
        self.mDoppler_2 = self.data.signals_mDoppler_2


    @classmethod
    def empty(cls):
        """
        Empty constructor
        """
        return cls(DataCutter.empty())
    
    
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
        processer = cls(DataCutter.empty())
        processer.load(path, file)
        return processer


    def padding(self,
                padding: int,
                mode: str='constant',
                **kwargs):
        """
        Pad the data.
        Check if the action is at least [padding] bins long, if not, pad the data.

        Parameters
        ----------
        padding : int
            Number of bins to pad.
        mode : str, optional
            Mode to use to pad the data. The default is 'constant'.
        **kwargs : TYPE
            Keyword arguments to pass to the pad function.
        """
        if self.do_rdn:
            self.padding_rdn(padding=padding, mode=mode, **kwargs)
        if self.do_mDoppler:
            self.padding_mDoppler(padding=padding, mode=mode, **kwargs)


    def padding_rdn(self,
                    padding: int=40,
                    mode: str='constant',
                    **kwargs):
        """
        Pad the rdn data.
        Check if the action is at least 40 bins long, if not, pad the data.

        Parameters
        ----------
        padding : int
            Number of bins to pad.
        mode : str, optional
            Mode to use to pad the data. The default is 'constant'.
        **kwargs : TYPE
            Keyword arguments to pass to the pad function.
        """
        if mode != 'last-frame':
            for i, (rdn_1, rdn_2) in enumerate(zip(self.rdn_1, self.rdn_2)):
                if rdn_1.shape[0] < padding:
                    self.rdn_1[i] = np.pad(
                        array=rdn_1,
                        pad_width=((0, padding - rdn_1.shape[0]), (0, 0), (0, 0)),
                        mode=mode,
                        **kwargs
                    )
                else:
                    self.rdn_1[i] = rdn_1
                if rdn_2.shape[0] < padding:
                    self.rdn_2[i] = np.pad(
                        array=rdn_2,
                        pad_width=((0, padding - rdn_2.shape[0]), (0, 0), (0, 0)),
                        mode=mode,
                        **kwargs
                    )
                else:
                    self.rdn_2[i] = rdn_2
        else:
            for i, (rdn_1, rdn_2) in enumerate(zip(self.rdn_1, self.rdn_2)):
                if rdn_1.shape[0] < padding:
                    last_frames_1 = rdn_1[-n:]  # Get the last n frames
                    padding_frames_1 = np.tile(last_frames_1, (padding - rdn_1.shape[0], 1))
                    self.rdn_1[i] = np.concatenate((rdn_1, padding_frames_1), axis=0)
                else:
                    self.rdn_1[i] = rdn_1
                if rdn_2.shape[0] < padding:
                    last_frames_2 = rdn_2[-n:]  # Get the last n frames
                    padding_frames_2 = np.tile(last_frames_2, (padding - rdn_2.shape[0], 1))
                    self.rdn_2[i] = np.concatenate((rdn_2, padding_frames_2), axis=0)
                else:
                    self.rdn_2[i] = rdn_2


    def padding_mDoppler(self,
                        padding: int,
                        n: int = 5,
                        mode: str = 'last-frame',
                        **kwargs):
        """
        Pad the mDoppler data.
        Check if the action is at least 40 bins long, if not, pad the data.

        Parameters
        ----------
        padding : int
            Number of bins to pad.
        n : int
            Number of last frames to repeat for padding.
        mode : str, optional
            Mode to use to pad the data. The default is 'constant'.
        **kwargs : TYPE
            Keyword arguments to pass to the pad function.
        """
        if mode != 'last-frame':
            for i, (mDoppler_1, mDoppler_2) in enumerate(zip(self.mDoppler_1, self.mDoppler_2)):
                if mDoppler_1.shape[0] < padding:
                    self.mDoppler_1[i] = np.pad(
                        array=mDoppler_1,
                        pad_width=((0, padding - mDoppler_1.shape[0]), (0, 0)),
                        mode=mode,
                        **kwargs
                    )
                else:
                    self.mDoppler_1[i] = mDoppler_1
                if mDoppler_2.shape[0] < padding:
                    self.mDoppler_2[i] = np.pad(
                        array=mDoppler_2,
                        pad_width=((0, padding - mDoppler_2.shape[0]), (0, 0)),
                        mode=mode,
                        **kwargs
                    )
                else:
                    self.mDoppler_2[i] = mDoppler_2
        else:
            for i, (mDoppler_1, mDoppler_2) in enumerate(zip(self.mDoppler_1, self.mDoppler_2)):
                if mDoppler_1.shape[0] < padding:
                    padding_length = padding - mDoppler_1.shape[0]
                    last_frames_1 = mDoppler_1[-n:]  # Get the last n frames
                    padding_frames_1 = np.tile(last_frames_1, (padding - mDoppler_1.shape[0], 1))[:padding_length]
                    self.mDoppler_1[i] = np.concatenate((mDoppler_1, padding_frames_1), axis=0)
                else:
                    self.mDoppler_1[i] = mDoppler_1
                if mDoppler_2.shape[0] < padding:
                    padding_length = padding - mDoppler_2.shape[0]
                    last_frames_2 = mDoppler_2[-n:]  # Get the last n frames
                    padding_frames_2 = np.tile(last_frames_2, (padding - mDoppler_2.shape[0], 1))[:padding_length]
                    self.mDoppler_2[i] = np.concatenate((mDoppler_2, padding_frames_2), axis=0)
                else:
                    self.mDoppler_2[i] = mDoppler_2


    def cut_time(self,
                 loc:str='random',
                 len_default:int=40,
                 n_samples:int=1,
                 **kwargs):
        """
        Cut the data in time

        Parameters
        ----------
        loc : str, optional
            Location of the cut. Possible values are:
            'center', 'start', 'end', 'random', 'normal',
            'threshold-center', 'threshold-start', 'threshold-end'.
            The default is 'random'.
        len_default : int, optional
            Default length of the action. The default is 40.
        n_samples : int, optional
            Number of samples to generate for each action. 
            Implemented only for random method. The default is 1.
        **kwargs : TYPE
            Keyword arguments to pass to the cut_time function.
        """
        if self.do_rdn:
            self.cut_time_rdn(loc=loc, len_default=len_default, n_samples=n_samples, **kwargs)
        if self.do_mDoppler:
            self.cut_time_mDoppler(loc=loc, len_default=len_default, n_samples=n_samples, **kwargs)


    def cut_time_rdn(self,
                     loc:int='random',
                     len_default:int=40,
                     n_samples:int=1,
                     **kwargs):
        """
        Cut the rdn data in time

        Parameters
        ----------
        loc : str, optional
            Location of the cut. Possible values are:
            'center', 'start', 'end', 'random', 'normal'.
            The default is 'random'.
        len_default : int, optional
            Default length of the action. The default is 40.
        n_samples : int, optional
            Number of samples to generate for each action.
            Implemented only for random method. The default is 1.
        **kwargs : TYPE
            Keyword arguments to pass to the cut_time function.

        Raises
        ------
        OptionIsFalseError
            If the option do_rdn is False.
        ValueError
            If the location is not valid.
        """
        if not self.do_rdn:
            raise OptionIsFalseError("do_rdn")

        for i, (rdn_1, rdn_2) in enumerate(zip(self.rdn_1, self.rdn_2)):
            if loc == 'center':
                self.rdn_1[i] = rdn_1[rdn_1.shape[0]//2-len_default//2:rdn_1.shape[0]//2+len_default//2, :]
                self.rdn_2[i] = rdn_2[rdn_2.shape[0]//2-len_default//2:rdn_2.shape[0]//2+len_default//2, :]
            elif loc == 'start':
                self.rdn_1[i] = rdn_1[:len_default, :]
                self.rdn_2[i] = rdn_2[:len_default, :]
            elif loc == 'end':
                self.rdn_1[i] = rdn_1[-len_default:, :]
                self.rdn_2[i] = rdn_2[-len_default:, :]
            elif loc == 'random':
                self.rdn_1[i] = cutting.random(rdn_1, len_default)
                self.rdn_2[i] = cutting.random(rdn_2, len_default)
            elif loc == 'normal':
                self.rdn_1[i] = cutting.normal(rdn_1, len_default, **kwargs)
                self.rdn_2[i] = cutting.normal(rdn_2, len_default, **kwargs)
            elif loc == 'threshold-start':
                self.rdn_1[i] = cutting.threshold_method(rdn_1, len_default, loc='start', **kwargs)
                self.rdn_2[i] = cutting.threshold_method(rdn_2, len_default, loc='start', **kwargs)
            elif loc == 'threshold-end':
                self.rdn_1[i] = cutting.threshold_method(rdn_1, len_default, loc='end', **kwargs)
                self.rdn_2[i] = cutting.threshold_method(rdn_2, len_default, loc='end', **kwargs)
            elif loc == 'threshold-center':
                self.rdn_1[i] = cutting.threshold_method(rdn_1, len_default, loc='center', **kwargs)
                self.rdn_2[i] = cutting.threshold_method(rdn_2, len_default, loc='center', **kwargs)
            elif loc == 'max-integral':
                self.mDoppler_1[i] = cutting.find_highest_integral_frames(mDoppler_1, len_default, **kwargs)
                self.mDoppler_2[i] = cutting.find_highest_integral_frames(mDoppler_2, len_default, **kwargs)
            else:
                raise ValueError("Invalid location")


    def cut_time_mDoppler(self,
                          loc:int='random',
                          len_default:int=40,
                          n_samples:int=1,
                          **kwargs):
        """
        Cut the mDoppler data in time

        Parameters
        ----------
        loc : str, optional
            Location of the cut. Possible values are:
            'center', 'start', 'end', 'random', 'normal',
            'threshold-center', 'threshold-start', 'threshold-end',
            'max-integral'.
            The default is 'random'.
        len_default : int, optional
            Default length of the action. The default is 40.
        n_samples : int, optional
            Number of samples to generate for each action. 
            Implemented only for random method. The default is 1.
        **kwargs : TYPE
            Keyword arguments to pass to the cut_time function.

        Raises
        ------
        OptionIsFalseError
            If do_mDoppler is False.
        ValueError
            If loc is not a valid value.
        """
        if not self.do_mDoppler:
            raise OptionIsFalseError("do_mDoppler")

        for i, (mDoppler_1, mDoppler_2) in enumerate(zip(self.mDoppler_1, self.mDoppler_2)):
            if loc == 'center':
                self.mDoppler_1[i] = mDoppler_1[mDoppler_1.shape[0]//2-len_default//2:mDoppler_1.shape[0]//2+len_default//2, :]
                self.mDoppler_2[i] = mDoppler_2[mDoppler_2.shape[0]//2-len_default//2:mDoppler_2.shape[0]//2+len_default//2, :]
            elif loc == 'start':
                self.mDoppler_1[i] = mDoppler_1[:len_default, :]
                self.mDoppler_2[i] = mDoppler_2[:len_default, :]
            elif loc == 'end':
                self.mDoppler_1[i] = mDoppler_1[-len_default:, :]
                self.mDoppler_2[i] = mDoppler_2[-len_default:, :]
            elif loc == 'random':
                self.mDoppler_1[i] = cutting.random(mDoppler_1, len_default)
                self.mDoppler_2[i] = cutting.random(mDoppler_2, len_default)
            elif loc == 'normal':
                self.mDoppler_1[i] = cutting.normal(mDoppler_1, len_default, **kwargs)
                self.mDoppler_2[i] = cutting.normal(mDoppler_2, len_default, **kwargs)
            elif loc == 'threshold-start':
                self.mDoppler_1[i] = cutting.threshold_method(mDoppler_1, len_default, loc='start', **kwargs)
                self.mDoppler_2[i] = cutting.threshold_method(mDoppler_2, len_default, loc='start', **kwargs)
            elif loc == 'threshold-end':
                self.mDoppler_1[i] = cutting.threshold_method(mDoppler_1, len_default, loc='end', **kwargs)
                self.mDoppler_2[i] = cutting.threshold_method(mDoppler_2, len_default, loc='end', **kwargs)
            elif loc == 'threshold-center':
                self.mDoppler_1[i] = cutting.threshold_method(mDoppler_1, len_default, loc='center', **kwargs)
                self.mDoppler_2[i] = cutting.threshold_method(mDoppler_2, len_default, loc='center', **kwargs)
            elif loc == 'max-integral':
                self.mDoppler_1[i] = cutting.find_highest_integral_frames(mDoppler_1, len_default, **kwargs)
                self.mDoppler_2[i] = cutting.find_highest_integral_frames(mDoppler_2, len_default, **kwargs)
            else:
                raise ValueError("Invalid location")


    def rotate(self):
        """
        Rotate the data.

        """
        if self.do_rdn:
            self.rotate_rdn()
        if self.do_mDoppler:
            self.rotate_mDoppler()
            
    
    def rotate_rdn(self):
        """
        Rotate the rdn data
        """
        
        if not self.do_rdn:
            raise OptionIsFalseError("do_rdn")
        
        for i, (rdn_1, rdn_2) in enumerate(zip(self.rdn_1, self.rdn_2)):
            self.rdn_1[i] = np.transpose(rdn_1)
            self.rdn_2[i] = np.transpose(rdn_2)


    def rotate_mDoppler(self):
        """
        Rotate the mDoppler data
        """
        
        if not self.do_mDoppler:
            raise OptionIsFalseError("do_mDoppler")
        
        for i, (mDoppler_1, mDoppler_2) in enumerate(zip(self.mDoppler_1, self.mDoppler_2)):
            self.mDoppler_1[i] = np.transpose(mDoppler_1)
            self.mDoppler_2[i] = np.transpose(mDoppler_2)


    def remove_low_mean(self,
                        threshold:float=0.1):
        """
        Remove the data with a low mean.
        
        Parameters
        ----------
        threshold : float, optional
            Threshold to remove the data. The default is 0.1.
        """
        if self.do_rdn:
            self.remove_low_mean_rdn(threshold)
        if self.do_mDoppler:
            self.remove_low_mean_mDoppler(threshold)
            
            
    def remove_low_mean_rdn(self,
                            threshold:float=0.1):
        """
        Remove the rdn data with a low mean.
        
        Parameters
        ----------
        threshold : float, optional
            Threshold to remove the data. The default is 0.1.
        """
        for i, (rdn_1, rdn_2) in enumerate(zip(self.rdn_1, self.rdn_2)):
            if np.mean(rdn_1) < threshold or np.mean(rdn_2) < threshold:
                self.rdn_1.pop(i)
                self.rdn_2.pop(i)
                self.labels.pop(i)
                self.labels_dict.pop(i)
                
    
    def remove_low_mean_mDoppler(self,
                                 threshold:float=0.1):
        """
        Remove the mDoppler data with a low mean.
        
        Parameters
        ----------
        threshold : float, optional
            Threshold to remove the data. The default is 0.1.
        """
        for i, (mDoppler_1, mDoppler_2) in enumerate(zip(self.mDoppler_1, self.mDoppler_2)):
            if np.mean(mDoppler_1) < threshold or np.mean(mDoppler_2) < threshold:
                self.mDoppler_1.pop(i)
                self.mDoppler_2.pop(i)
                self.labels.pop(i)
                self.labels_dict.pop(i)


    def save(self,
             path:str='DATA_preprocessed',
             filename:str='data_processed.npz'):
        """
        Save the data.
        
        Parameters
        ----------
        path : str, optional
            Path to the file, by default 'DATA_preprocessed'
        filename : str, optional
            Name of the file, by default 'data_processed.npz'
        """

        # Create the path
        path = os.path.join(path, filename)

        # Save the data
        np.savez(
            path,
            mDoppler_1=self.mDoppler_1,
            mDoppler_2=self.mDoppler_2,
            rdn_1=self.rdn_1,
            rdn_2=self.rdn_2,
            labels=self.labels,
            #labels_dict=self.labels_dict
        )


    def load(self,
             path:str='DATA_preprocessed',
             filename:str='data_processed.npz'):
        """
        Load the data from npz file.
        
        Parameters
        ----------
        path : str, optional
            Path to the file, by default 'DATA_preprocessed'
        filename : str, optional
            Name of the file, by default 'data_processed.npz'
        """

        # Create the path
        path = os.path.join(path, filename)

        # Load the data
        data = np.load(path, allow_pickle=True)

        # Get the data
        self.mDoppler_1  = data['mDoppler_1']
        self.mDoppler_2  = data['mDoppler_2']
        self.rdn_1       = data['rdn_1']
        self.rdn_2       = data['rdn_2']
        self.labels      = data['labels']
        #self.labels_dict = data['labels_dict'].item()

        if self.mDoppler_1 is None:
            self.do_mDoppler = False
        if self.rdn_1 is None:
            self.do_rdn = False