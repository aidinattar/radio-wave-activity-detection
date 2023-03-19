### TO BE CORRECTED: I removed the data reader part, so this code will not work
### as it is. I will adapt it to the new data reader class.

'''
Read the data from the h5 files and perform clustering on the data
'''

import os
import cv2
import hdbscan
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

from tqdm                 import tqdm
from matplotlib           import animation
from sklearn.cluster      import DBSCAN, KMeans, \
                                 AgglomerativeClustering
from sklearn.mixture      import GaussianMixture
from sklearn.metrics      import make_scorer
from utils                import my_silhouette_score, \
                                 pipeline
from DataReader           import OptionIsFalseError

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score

data_dir    = os.path.join(os.getcwd(),    'DATA')
figures_dir = os.path.join(os.getcwd(), 'figures')


class preprocess(object):
    '''
    Class to preprocess the data
    '''

    def __init__(self):
        pass

    def dbscan(self, eps=3, min_samples=10, name='dbscan.png', save=False):
        '''
        Function to perform the DBSCAN clustering algorithm
        '''

        if not self.do_mDoppler:
            raise OptionIsFalseError("do_mDoppler")

        # Get the first image
        img = self.mDoppler[0].T

        # Reshape the image
        img = img.reshape((img.shape[0] * img.shape[1], 1))

        # Perform DBSCAN
        db = DBSCAN(eps=3, min_samples=10).fit(img)

        # Create an empty array
        labels = np.zeros_like(db.labels_)
        labels[db.labels_ != -1] = 1

        # Reshape the labels
        labels = labels.reshape((self.mDoppler[0].shape[0], self.mDoppler[0].shape[1]))

        # Plot the results
        fig, ax = plt.subplots(nrows=2, figsize=(9,7))

        ax[0].imshow(self.mDoppler[0], cmap='viridis')
        ax[0].set_title("Original")

        ax[1].imshow(labels, cmap='viridis')
        ax[1].set_title("DBSCAN")

        fig.tight_layout()
        plt.show()

        if save:
            fig.savefig(os.path.join(figures_dir, name))


    def hdbscan(self, name='hdbscan.png', save=False, **kwargs):
        '''
        Function to perform the HDBSCAN clustering algorithm
        '''

        if not self.do_mDoppler:
            raise OptionIsFalseError("do_mDoppler")

        # Get the first image
        img = self.mDoppler[0].T

        # Reshape the image
        img = img.reshape((img.shape[0] * img.shape[1], 1))

        # Perform HDBSCAN
        clusterer = hdbscan.HDBSCAN(**kwargs)
        clusterer.fit(img)

        # Create an empty array
        labels = np.zeros_like(clusterer.labels_)
        labels[clusterer.labels_ != -1] = 1

        # Reshape the labels
        labels = labels.reshape((self.mDoppler[0].shape[0], self.mDoppler[0].shape[1]))

        #labels = pipeline(self.mDoppler[0].T, **kwargs)

        # Plot the results
        fig, ax = plt.subplots(nrows=2, figsize=(9,7))

        ax[0].contourf(self.mDoppler[0].T, cmap='viridis')
        ax[0].set_title("Original")

        ax[1].contourf(labels, cmap='viridis')
        ax[1].set_title("HDBSCAN")

        fig.tight_layout()
        plt.show()

        if save:
            fig.savefig(os.path.join(figures_dir, name))


    def kmeans(self, n_clusters=112, name='kmeans.png', save=False):
        '''
        Function to perform the K-means clustering algorithm
        '''

        if not self.do_mDoppler:
            raise OptionIsFalseError("do_mDoppler")

        # Get the first image
        img = self.mDoppler[0].T

        # Perform K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(img.reshape(-1, 1))

        # Get the labels and reshape them
        labels = kmeans.labels_.reshape(img.shape)

        # Plot the results
        fig, ax = plt.subplots(nrows=2, figsize=(9,7))

        ax[0].imshow(img, cmap='viridis', aspect=20)
        ax[0].set_title("Original")

        ax[1].imshow(labels, cmap='viridis', aspect=20)
        ax[1].set_title("K-means")

        fig.tight_layout()
        plt.show()

        if save:
            fig.savefig(os.path.join(figures_dir, name))



    def xmeans(self, name='xmeans.png', save=False):
        '''
        Function to perform the X-means clustering algorithm
        '''

        if not self.do_mDoppler:
            raise OptionIsFalseError("do_mDoppler")

        # Get the first image
        img = self.mDoppler[0].T

        # Reshape the image
        img = img.reshape((img.shape[0] * img.shape[1], 1))

        # Perform X-means
        xmeans = XMeans(random_state=0).fit(img)

        # Create an empty array
        labels = np.zeros_like(xmeans.labels_)
        labels[xmeans.labels_ != -1] = 1

        # Reshape the labels
        labels = labels.reshape((self.mDoppler[0].shape[0], self.mDoppler[0].shape[1]))

        # Plot the results
        fig, ax = plt.subplots(nrows=2, figsize=(9,7))

        ax[0].imshow(self.mDoppler[0], cmap='viridis')
        ax[0].set_title("Original")

        ax[1].imshow(labels, cmap='viridis')
        ax[1].set_title("X-means")

        fig.tight_layout()
        plt.show()

        if save:
            fig.savefig(os.path.join(figures_dir, name))


    def hierarchical_clustering(self, name='hierarchical_clustering.png', save=False):
        '''
        Function to perform the hierarchical clustering algorithm
        '''

        if not self.do_mDoppler:
            raise OptionIsFalseError("do_mDoppler")

        # Get the first image
        img = self.mDoppler[0].T

        # Reshape the image
        img = img.reshape((img.shape[0] * img.shape[1], 1))

        # Perform hierarchical clustering
        clustering = AgglomerativeClustering().fit(img)

        # Create an empty array
        labels = np.zeros_like(clustering.labels_)
        labels[clustering.labels_ != -1] = 1

        # Reshape the labels
        labels = labels.reshape((self.mDoppler[0].shape[0], self.mDoppler[0].shape[1]))

        # Plot the results
        fig, ax = plt.subplots(nrows=2, figsize=(9,7))

        ax[0].imshow(self.mDoppler[0], cmap='viridis')
        ax[0].set_title("Original")

        ax[1].imshow(labels, cmap='viridis')
        ax[1].set_title("Hierarchical clustering")

        fig.tight_layout()
        plt.show()

        if save:
            fig.savefig(os.path.join(figures_dir, name))

    def GMM(self, name='GMM.png', save=False):
        '''
        Function to perform the GMM clustering algorithm
        '''

        if not self.do_mDoppler:
            raise OptionIsFalseError("do_mDoppler")

        # Get the first image
        img = self.mDoppler[0].T

        # Reshape the image
        img = img.reshape((img.shape[0] * img.shape[1], 1))

        # Perform GMM
        gmm = GaussianMixture(n_components=3, random_state=0).fit(img)

        # Create an empty array
        labels = np.zeros_like(gmm.predict(img))
        labels[gmm.predict(img) != -1] = 1

        # Reshape the labels
        labels = labels.reshape((self.mDoppler[0].shape[0], self.mDoppler[0].shape[1]))

        # Plot the results
        fig, ax = plt.subplots(nrows=2, figsize=(9,7))

        ax[0].contourf(self.mDoppler[0], cmap='viridis')
        ax[0].set_title("Original")

        ax[1].contourf(labels, cmap='viridis')
        ax[1].set_title("GMM")

        fig.tight_layout()
        plt.show()

        if save:
            fig.savefig(os.path.join(figures_dir, name))


    def gridsearch(self):
        '''
        Function to perform a grid search to find the best parameters for the
        HDBSCAN algorithm
        '''

        if not self.do_mDoppler:
            raise OptionIsFalseError("do_mDoppler")

        # Get the first image
        img = self.mDoppler[0].T

        # Reshape the image
        img = img.reshape((img.shape[0] * img.shape[1], 1))


        # Define the parameters to search over
        param_grid = {'min_cluster_size': [5, 10, 15], 
                    'min_samples': [5, 10, 15],
                    'alpha': [0.5, 1.0],
                    'metric': ['euclidean', 'manhattan']}

        # Perform grid search
        clusterer = hdbscan.HDBSCAN()

        # Create the grid search object
        grid_search = GridSearchCV(clusterer, param_grid, cv=5, scoring=my_silhouette_score, verbose=3)

        grid_search.fit(img)

        # Print the best parameters and score
        print("Best parameters: ", grid_search.best_params_)
        print("Best score: ", grid_search.best_score_)