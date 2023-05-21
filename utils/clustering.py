"""
clustering.py

Classes and functions for clustering the data.
"""

import cv2
import numpy as np
import scipy.ndimage as ndi
import skimage.morphology as morph
import skimage.measure as measure
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def my_silhouette_score(model, X, y=None):
    preds = model.fit_predict(X)
    return silhouette_score(X, preds) if len(set(preds)) > 1 else float('nan')

def preprocess(img, kernel_size=3):
    # Apply Gaussian smoothing to reduce noise
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return img

def threshold(img):
    # Use Otsu's method to determine threshold value
    _, thresholded = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded

def morphological_ops(thresholded, erosion_size=3, dilation_size=3):
    # Remove small noise pixels using erosion
    eroded = morph.erosion(thresholded, morph.square(erosion_size))

    # Fill in gaps in signal clouds using dilation
    dilated = morph.dilation(eroded, morph.square(dilation_size))
    return dilated

def connected_components(dilated):
    # Label individual signal clouds with different indices
    labeled, num_features = ndi.label(dilated)
    return labeled, num_features

def feature_extraction(labeled):
    # Extract size, shape, and intensity distribution features
    properties = measure.regionprops(labeled, intensity_image=labeled)
    features = [(prop.area, prop.eccentricity, np.mean(prop.intensity_image)) for prop in properties]
    return features

def clustering(features, num_clusters=3):
    # Cluster signal clouds based on their features
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)
    return kmeans.labels_

def pipeline(img, kernel_size=3, erosion_size=3, dilation_size=3, num_clusters=3):
    # Preprocess image
    img_processed = preprocess(img, kernel_size)

    # Apply thresholding
    thresholded = threshold(img_processed)

    # Apply morphological operations
    dilated = morphological_ops(thresholded, erosion_size, dilation_size)

    # Perform connected component analysis
    labeled, num_features = connected_components(dilated)

    # Extract features
    features = feature_extraction(labeled)

    # Cluster signal clouds
    labels = clustering(features, num_clusters)

    return labels