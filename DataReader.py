'''
DataReader.py

This file contains the DataReader class,
which is used to read data contained in the directory DATA.
'''

import os
import h5py
import cv2
import copy
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

from tqdm                 import tqdm
from matplotlib           import animation
from utils                import rescaling

data_dir    = os.path.join(os.getcwd(),    'DATA')
figures_dir = os.path.join(os.getcwd(), 'figures')

class OptionIsFalseError(Exception):
    def __init__(self, option_name):
        self.option_name = option_name

    def __str__(self):
        return f"The '{self.option_name}' option is set to False. The code cannot be run."


class DataReader(object):
    '''
    Class to read the data and do some prelimary processing
    '''

    # Flags to indicate if the data has been processed
    timestamp_to_bins_done = False
    radar_division_done    = False

    def __init__(self, subjects, sets, do_rdn:bool=False, do_mDoppler:bool=True):
        '''
        Initialize the class

        Parameters
        ----------
        subjects : list
            List of subjects to process
        sets : list
            List of sets to process
        rdn : bool, optional
            Flag to indicate if rdn data should be processed [default: False]
        mDoppler : bool, optional
            Flag to indicate if mDoppler data should be processed [default: True]
        '''
        os.chdir(data_dir)

        self.subjects      = subjects
        self.sets          = sets
        self.captured_data = 'captured_data'
        self.files         = ['timestamp_speech.csv', 'processed_data.h5']
        self.do_rdn        = do_rdn
        self.do_mDoppler   = do_mDoppler

        self.rdn, self.mDoppler, self.timestamp_speech = self.ReadData()


    def ReadData(self):
        '''
        Function to read the data from the h5 files
        '''

        # create lists to store the data
        rdn, mDoppler, timestamp_speech = ([] for i in range(3))

        # loop over each subject
        for subject in tqdm(self.subjects):

            # change the current working directory to the new folder
            os.chdir(os.path.join(subject, self.captured_data))

            # loop over each set_
            for set_ in self.sets:
                if (subject == 'subject_09') and (set_ == 'set000'):
                    continue
                try:
                    os.chdir(set_)
                    for file in self.files:
                        if '.h5' in file:
                            # Open the file in read-only mode
                            with h5py.File(file, 'r') as f:
                                # Access a group and list its contents
                                group = f['radar']
                                for key1 in group.keys():
                                    group_group = group[key1]
                                    if self.do_rdn:
                                        rdn.append(group_group['rdn'][()])
                                    if self.do_mDoppler:
                                        mDoppler.append(group_group['mDoppler'][()])
                        else:
                            timestamp_speech.append(pd.read_csv(file))
                            pass
                except FileNotFoundError:
                    pass
                os.chdir('..')
            os.chdir('..')
            os.chdir('..')

        os.chdir('..')

        return rdn, mDoppler, timestamp_speech


    def remove_static_bins(self):
        '''
        Remove the static bins from the data
        '''

        for i in range(len(self.rdn)):
            self.rdn[i] = np.delete(self.rdn[i], (63,64,65), axis=2)

        for i in range(len(self.rdn)):
            self.mDoppler[i] = np.delete(self.mDoppler[i], (63,64,65), axis=1)


    def crop_rdn(self,
                 start_vel:int=13, stop_vel:int=111,
                 start_range:int=0, stop_range:int=63):
        '''
        Crop the rdn data

        Parameters
        ----------
        start_vel : int, optional
            Start velocity bin [default: 13]
        stop_vel : int, optional
            Stop velocity bin [default: 111]
        start_range : int, optional
            Start range bin [default: 0]
        stop_range : int, optional
            Stop range bin [default: 63]
        '''
        # check if the dimensions are correct
        if not self.do_rdn:
            raise OptionIsFalseError('do_rdn')

        for i in range(len(self.rdn)):
            self.rdn[i] = self.rdn[i][start_vel:stop_vel, start_range:stop_range, :]


    def crop_mDoppler(self, start: int, stop: int):
        '''
        Crop the mDoppler data
        '''
        # check if the dimensions are correct
        if not self.do_mDoppler:
            raise OptionIsFalseError('do_mDoppler')

        for i in range(len(self.mDoppler)):
            self.mDoppler[i] = self.mDoppler[i][start:stop, :]


    def radar_division(self):
        '''
        Divide the data based on the radar they come from
        '''

        if self.do_mDoppler:
            self.mDoppler_1 = self.mDoppler[0::2]
            self.mDoppler_2 = self.mDoppler[1::2]

        if self.do_rdn:
            self.rdn_1 = self.rdn[0::2]
            self.rdn_2 = self.rdn[1::2]

        self.radar_division_done = True


    def rescaling(self, method:str='norm', **kwargs):
        '''
        Function to rescale the data

        Parameters
        ----------
        method : str, optional
            Method to use for rescaling, possible methods:
                - 'norm': normalize the data
                - 'std': standardize the data
                - 'log': log transformation
                - 'pow': power transformation
                - 'bin': binning
                - 'pca': apply pca
                - 'min_max': min max
            [default: 'norm']
        kwargs : dict
            Dictionary of keyword arguments for the rescaling method
        '''

        method_dict = {
            'norm':    rescaling.normalize,
            'std':     rescaling.standardize,
            'log':     rescaling.log_transformation,
            'pow':     rescaling.power_transformation,
            'bin':     rescaling.binning,
            'pca':     rescaling.apply_pca,
            'min_max': rescaling.min_max
        }

        if method not in method_dict:
            raise ValueError("Invalid method")

        if self.do_mDoppler:
            for i in range(len(self.mDoppler)):
                self.mDoppler[i] = method_dict[method](data=self.mDoppler[i], **kwargs)

        if self.do_rdn:
            for i in range(len(self.rdn)):
                self.rdn[i] = method_dict[method](data=self.rdn[i], **kwargs)


    def Plot_Gif_rdn(self, frames:int, k:int=0, name:str='rdn.gif'):
        '''
        Function to plot the gif of the rdn data

        Parameters
        ----------
        frames : int
            Number of frames to plot
        k : int
            Index of the radar to plot [default: 0]
        name : str, optional
            Name of the file to save [default: 'rdn.gif']
        '''
        def update(j:int):
            for i in range(2*k, 2*k+2):
                data = self.rdn[i][j,:,:]

                # Clear previous plot
                ax[i].clear()

                # Create a contour plot
                ax[i].contourf(data)

                # Add x and y labels
                ax[i].set_xlabel("velocity [m/s]")
                ax[i].set_ylabel("range [m]")

            fig.suptitle("Frame {}".format(j))
            fig.tight_layout()
            # Pause between frames
            plt.pause(0.1)

        if not self.do_rdn:
            raise OptionIsFalseError("do_rdn")

        fig, ax = plt.subplots(ncols=2, figsize=(7,9))

        # Create the animation
        anim = animation.FuncAnimation(fig, update, frames=frames, interval=50)

        # Save the animation as a GIF file
        anim.save(os.path.join(figures_dir, name), writer="pillow")

    def plot_rdn_map(self, k:int=0, range_length:int=40, name:str='rdn_map.png', save:bool=False):
        '''
        Function to plot the map of the rdn data

        Parameters
        ----------
        k : int
            Index of the radar to plot [default: 0]
        range_length : int
            Number of frames to consider [default: 40]
        name : str, optional
            Name of the file to save [default: 'rdn_map.png']
        '''
        if not self.do_rdn:
            raise OptionIsFalseError("do_rdn")

        fig, ax = plt.subplots(ncols=2, figsize=(7,9))

        for i in range(2*k, 2*k+2):
            data = self.rdn[i][int(k*range_length):int((k+1)*range_length),:,:].max(axis=0)

            # Create a contour plot
            ax[i].contourf(data)

            # Add x and y labels
            ax[i].set_xlabel("velocity [m/s]")
            ax[i].set_ylabel("range [m]")

        fig.tight_layout()
        # Show the plot
        plt.show()

        if save:
            fig.savefig(os.path.join(figures_dir, name))

    def plot_rdn_map_3d(self, k=0, range_length=40, name='rdn_map_3d.png', save=False):
        '''
        Function to plot the map of the rdn data

        Parameters
        ----------
        k : int
            Index of the radar to plot [default: 0]
        range_length : int
            Number of frames to consider [default: 40]
        name : str, optional
            Name of the file to save [default: 'rdn_map_3d.png']
        '''
        if not self.do_rdn:
            raise OptionIsFalseError("do_rdn")

        fig, ax = plt.subplots(ncols=2, subplot_kw={'projection': '3d'})

        for i in range(2*k, 2*k+2):
            data = self.rdn[i][int(k*range_length):int((k+1)*range_length),:,:].max(axis=0)

            # Create a grid of X and Y coordinates
            x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

            # Create a 3D contour plot
            ax[i].plot_surface(x, y, data, cmap='viridis')

            # Create a contour plot
            #ax[i].contourf(x, y, data, zdir='z', offset=-2, cmap='coolwarm')

            # Add x and y labels
            ax[i].set_xlabel("velocity [m/s]")
            ax[i].set_ylabel("range [m]")

        fig.tight_layout()
        # Show the plot
        plt.show()

        if save:
            fig.savefig(os.path.join(figures_dir, name))

    def Plot_Gif_mDoppler(self, frames, k=0, name='mDoppler.gif'):
        '''
        Function to plot the gif of the mDoppler data

        Parameters
        ----------
        frames : list
            List of frames to plot
        k : int
            Index of the radar to plot [default: 0]
        name : str, optional
            Name of the file to save [default: 'mDoppler.gif']
        '''
        length_range = len(self.mDoppler[2*k]) / frames

        def update(j):
            for i in range(2*k, 2*k+2):
                data = self.mDoppler[i][int(j*length_range):int((j+1)*length_range)].T

                # Clear previous plot
                ax[i].clear()

                # Create a contour plot
                ax[i].contourf(data)

                # Add x and y labels
                ax[i].set_xlabel("time [s]")
                ax[i].set_ylabel("velocity [m/s]")

            fig.suptitle("Frame {}".format(j))
            fig.tight_layout()
            # Pause between frames
            plt.pause(0.1)

        if not self.do_mDoppler:
            raise OptionIsFalseError("do_mDoppler")

        fig, ax = plt.subplots(nrows=2, figsize=(9,7))

        # Create the animation
        anim = animation.FuncAnimation(fig, update, frames=frames, interval=100)

        # Save the animation as a GIF file
        anim.save(os.path.join(figures_dir, name), writer="pillow")

    def plot_mDoppler_map(self, k=0, start=10, stop=90, name='mDoppler_map.png', save=False):
        '''
        Function to plot the map of the mDoppler data

        Parameters
        ----------
        k : int
            Index of the radar to plot [default: 0]
        start : int
            Start of the range [default: 10]
        stop : int
            End of the range [default: 90]
        name : str, optional
            Name of the file to save [default: 'mDoppler_map.png']
        save : bool, optional
            Save the figure [default: False]
        '''
        if not self.do_mDoppler:
            raise OptionIsFalseError("do_mDoppler")

        fig, ax = plt.subplots(nrows=2, figsize=(9,7))

        for i in range(2*k, 2*k+2):
            data = self.mDoppler[i][start:stop].T

            # Create a contour plot
            ax[i].contourf(data)

            # Add x and y labels
            ax[i].set_xlabel("time [au]")
            ax[i].set_ylabel("velocity [m/s]")

        fig.tight_layout()
        # Show the plot
        plt.show()

        if save:
            fig.savefig(os.path.join(figures_dir, name))

    def plot_mDoppler_map_3d(self, k=0, start=10, stop=90, name='mDoppler_map_3d.png', save=False):
        '''
        TO CORRECT
        Function to plot the map of the mDoppler data

        Parameters
        ----------
        k : int
            Index of the radar to plot [default: 0]
        start : int
            Start of the range [default: 10]
        stop : int
            End of the range [default: 90]
        name : str, optional
            Name of the file to save [default: 'mDoppler_map_3d.png']
        save : bool, optional
            Save the figure [default: False]
        '''
        if not self.do_mDoppler:
            raise OptionIsFalseError("do_mDoppler")

        fig, ax = plt.subplots(ncols=2, subplot_kw={'projection': '3d'})

        for i in range(2*k, 2*k+2):
            data = self.mDoppler[i][start:stop].T

            # Create a grid of X and Y coordinates
            x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

            # Create a 3D contour plot
            #ax[i].plot_surface(x, y, data, cmap='viridis')

            # Create a contour plot
            ax[i].contourf(x, y, data, zdir='z', offset=-2, cmap='coolwarm')

            # Add x and y labels
            ax[i].set_xlabel("time [au]")
            ax[i].set_ylabel("velocity [m/s]")

        fig.tight_layout()
        # Show the plot
        plt.show()

        if save:
            fig.savefig(os.path.join(figures_dir, name))

    def filter_mDoppler(self, size=(5,5), sigma=1, name='mDoppler_filtered.png', save=False):
        '''
        Function to filter the mDoppler data

        Parameters
        ----------
        size : tuple
            Size of the filter [default: (5,5)]
        sigma : float
            Sigma of the filter [default: 1]
        name : str, optional
            Name of the file to save [default: 'mDoppler_filtered.png']
        save : bool, optional
            Save the figure [default: False]
        '''

        if not self.do_mDoppler:
            raise OptionIsFalseError("do_mDoppler")

        # Filter the data
        self.mDoppler = [cv2.GaussianBlur(data, size, 1) for data in self.mDoppler]


    def timestamp_to_bins(self, conversion_factor:float=1):
        '''
        Function to convert the timestamps in bins

        Parameters
        ----------
        conversion_factor : float
            Conversion factor to convert the timestamps in seconds [default: 1]
        '''
        # Compute the time passed
        for i in range(len(self.timestamp_speech)):
            # Convert the timestamps in seconds
            self.timestamp_speech[i]['timestamp']=pd.to_datetime(self.timestamp_speech[i]['timestamp'])
            # Compute the time passed
            self.timestamp_speech[i]['time_passed'] = (self.timestamp_speech[i]['timestamp'] - self.timestamp_speech[i]['timestamp'].iloc[0]).dt.total_seconds()
            # compute the time passed in bins
            self.timestamp_speech[i]['time_passed_bins'] = (self.timestamp_speech[i]['time_passed']*conversion_factor).astype(int)
            # compute number of beans per action
            self.timestamp_speech[i]['n_beans'] = self.timestamp_speech[i]['time_passed_bins'].diff().fillna(0).astype(int)

        self.timestamp_to_bins_done = True


    def plot_divided_actions(self, k:int=0, name:str='actions.png', save:bool=False):
        '''
        Function to divide the actions according to the
        timestamps recorded in the file

        Parameters
        ----------
        k : int
            Index of the radar to plot [default: 0]
        name : str, optional
            Name of the file to save [default: 'actions.png']
        save : bool, optional
            Save the figure [default: False]
        '''

        if not self.timestamp_to_bins_done:
            raise OptionIsFalseError("timestamp_to_bins")
        if not self.do_mDoppler:
            raise OptionIsFalseError("do_mDoppler")

        # define the figure
        fig, axes = plt.subplots(nrows=2, figsize=(9,7))

        # plot the data
        for i, ax in enumerate(axes):
            ax.contourf(self.mDoppler[k+i].T, cmap='coolwarm')
            # draw vertical lines corresponding to the actions' limits
            for j in range(len(self.timestamp_speech[k])):
                ax.axvline(x=self.timestamp_speech[k]['time_passed_bins'].iloc[j], color='red', linestyle='--')

        plt.show()

        if save:
            fig.savefig(os.path.join(figures_dir, name))

    def copy(self):
        '''
        Function to copy the object
        '''
        return copy.deepcopy(self)