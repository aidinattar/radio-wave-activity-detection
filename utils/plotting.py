"""
plotting.py

Classes and functions for plotting the results
"""

import os
import pandas            as pd
import matplotlib.pyplot as plt
import seaborn           as sns

sns.set_style('darkgrid')
class PlotHistory:
    """
    Plot the history of the training
    """
    def __init__(self, history: pd.DataFrame):
        """
        Constructor

        Parameters
        ----------
        history : pd.DataFrame
            History of the training
        """
        self.history = history

    def plot(self, show=True, save=False, path='./', name='plot.png'):
        """
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
        """
        # Set font size
        plt.rcParams.update({'font.size': 14})

        # Set figure size
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))

        for AX, metric in zip(ax, ['acc', 'loss']):
            # Set title and labels
            AX.set_title('Loss' if metric == 'loss' else 'Accuracy', fontsize=16)
            AX.set_xlabel('Epoch', fontsize=14)
            AX.set_ylabel('Loss' if metric == 'loss' else 'Accuracy', fontsize=14)

            # Plot the history
            AX.plot(self.history[f'test_{metric}'], label=f'Validation {metric}', color='#00ff00')
            
            if 'train_loss' in self.history.columns:
                AX.plot(self.history[f'train_{metric}'], label=f'Training {metric}', color='#ff00ff')
            
            AX.legend()

            # Add grid
            AX.grid(True)

        # Adjust margins
        #plt.subplots_adjust(wspace=0.3)
        fig.tight_layout()

        # Show the plot
        if show:
            plt.show()

        # Save the plot
        if save:
            if not os.path.exists(path):
                os.makedirs(path)
            fig.savefig(os.path.join(path, name))
