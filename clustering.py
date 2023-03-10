'''
Read the data from the h5 files and perform clustering on the data
'''

import os
import h5py
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

from tqdm                 import tqdm
from matplotlib           import animation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster      import DBSCAN
from rescaling            import normalize, standardize, \
                                 log_transformation, power_transformation, \
                                 binning, apply_pca

data_dir    = os.path.join(os.getcwd(),    'DATA')
figures_dir = os.path.join(os.getcwd(), 'figures')


class OptionIsFalseError(Exception):
    def __init__(self, option_name):
        self.option_name = option_name

    def __str__(self):
        return f"The '{self.option_name}' option is set to False. The code cannot be run."


class preprocess(object):
    '''
    Class to preprocess the data
    '''

    def __init__(self, subjects, sets, do_rdn=False, do_mDoppler=True):
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

    def rescaling(self, method='norm', **kwargs):
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
            [default: 'norm']
        kwargs : dict
            Dictionary of keyword arguments for the rescaling method
        '''

        method_dict = {
            'norm': normalize,
            'std': standardize,
            'log': log_transformation,
            'pow': power_transformation,
            'bin': binning,
            'pca': apply_pca
        }

        if method not in method_dict:
            raise ValueError("Invalid method")

        if self.do_mDoppler:
            for i in range(len(self.mDoppler)):
                self.mDoppler[i] = method_dict[method](data=self.mDoppler[i], **kwargs)

        if self.do_rdn:
            for i in range(len(self.rdn)):
                self.rdn[i] = method_dict[method](data=self.rdn[i], **kwargs)


    def Plot_Gif_rdn(self, frames, k=0, name='rdn.gif'):
        '''
        Function to plot the gif of the rdn data

        Parameters
        ----------
        frames : list
            List of frames to plot
        k : int
            Index of the radar to plot [default: 0]
        name : str, optional
            Name of the file to save [default: 'rdn.gif']
        '''
        def update(j):
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

    def plot_rdn_map(self, k=0, range_length=40, name='rdn_map.png', save=False):
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

    def dbscan(self):
        pass

    def kmeans(self):
        pass

    def xmeans(self):
        pass

    def hierarchical_clustering(self):
        pass

    def GMM(self):
        pass

    def plot_clusters(self):
        pass