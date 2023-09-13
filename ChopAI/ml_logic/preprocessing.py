import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import os
import sys

#################################

def create_and_preprocess_X_y():
    """
    Create X and y arrays of respective shape (88, 7400, 106) and (88, 100, 106) from music images, concatenated from each folder
    """

    folder_list = [folder for folder in os.listdir("../../data_image")]

    len_max = 0  # Maximum number of images per music piece in dataset
    for folder in folder_list:
        image_list = [image for image in os.listdir(f"../../data_image/{folder}")]
        if len(image_list) > len_max:
            len_max = len(image_list)

    X = np.zeros((len(folder_list), (len_max - 1) * 100, 106), dtype=float)
    y = np.zeros((len(folder_list), 100, 106), dtype=float)

    for index_folder, folder in enumerate(folder_list):
        image_list = [image for image in os.listdir(f"../../data_image/{folder}")]
        nb_images = len(image_list)
        folder_X = np.full(((len_max - 1) * 100, 106), -1, dtype=float)
        folder_y = np.full((100, 106), -1, dtype=float)

        for index_image, image in enumerate(image_list):
            image_array = np.transpose(plt.imread(f"../../data_image/{folder}/{image}"))
            if index_image < (nb_images - 1):
                folder_X[index_image * 100: (index_image + 1) * 100, :] = image_array
            elif index_image == (nb_images - 1):
                folder_y = image_array

            X[index_folder, :, :] = folder_X
            y[index_folder, :, :] = folder_y

    return X, y

#################################

def create_train_test_set(X, y, train_size):
    """
    Create train and test sets from X and y:
        - Train set contains first (train_size)% music pieces
        - Test set contains last (1-train_size)% music pieces
    """
    total_samples = X.shape[0]
    train_samples = int(total_samples * train_size)
    X_train, y_train = X[:train_samples,:,:], y[:train_samples,:,:]
    X_test, y_test = X[train_samples:,:,:], y[train_samples:,:,:]

    return X_train, X_test, y_train, y_test
