'''
plotting.py

Classes and functions for plotting the results
'''

import os
import pandas            as pd
import matplotlib.pyplot as plt

class PlotHistory:
    '''
    Plot the history of the training
    '''
    def __init__(self, history: pd.DataFrame):
        '''
        Constructor

        Parameters
        ----------
        history : pd.DataFrame
            History of the training
        '''
        self.history = history

    def plot(self, show=True, save=False, path='./', name='plot.png'):
        '''
        Plot the history

        Parameters
        ----------
        show : bool, optional
            Show the plot. The default is True.
        save : bool, optional
            Save the plot. The default is False.
        path : str, optional
            Path to the folder where to save the plot. The default is './'.
        name : str, optional
            Name of the plot. The default is 'plot.png'.
        '''
        plt.style.use('dark_background')

        # Set font size
        plt.rcParams.update({'font.size': 14})

        # Set figure size
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))

        # Set title and labels
        ax[0].set_title('Loss', fontsize=16)
        ax[0].set_xlabel('Epoch', fontsize=14)
        ax[0].set_ylabel('Loss', fontsize=14)

        ax[1].set_title('Accuracy', fontsize=16)
        ax[1].set_xlabel('Epoch', fontsize=14)
        ax[1].set_ylabel('Accuracy', fontsize=14)

        # Plot the history
        ax[0].plot(self.history['loss'], label='Training loss', color='#ff00ff')
        ax[0].plot(self.history['val_loss'], label='Validation loss', color='#00ffff')
        ax[0].legend()

        ax[1].plot(self.history['accuracy'], label='Training accuracy', color='#ff00ff')
        ax[1].plot(self.history['val_accuracy'], label='Validation accuracy', color='#00ffff')
        ax[1].legend()

        # Add grid
        ax[0].grid(True)
        ax[1].grid(True)

        # Adjust margins
        plt.subplots_adjust(wspace=0.3)
        #fig.tight_layout()

        # Show the plot
        if show:
            plt.show()

        # Save the plot
        if save:
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(os.path.join(path, name))
