# Data manipulation
import numpy as np
import pandas as pd

# Data Visualiation
import matplotlib.pyplot as plt
import seaborn as sns

# System
import os

# Performance metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Package imports
from preprocessing import create_and_preprocess_X_y, create_train_test_set

#################################

def create_y_pred_baseline(train_size):
    """
    Based on last seen sequence, get:
        - y_train_pred: predictions for the train set
        - y_test_pred: predictions for the test set
    Use same train_size as in ml_logic/preprocessing/create_train_test_set to split y_pred with same ratio
    """

    folder_list = [folder for folder in os.listdir("../../data_image")]

    y_pred = np.zeros((len(folder_list), 100, 106), dtype=float)

    for index_folder, folder in enumerate(folder_list):
        image_list = [image for image in os.listdir(f"../../data_image/{folder}")]
        nb_images = len(image_list)

        for index_image, image in enumerate(image_list):
            image_array = np.transpose(plt.imread(f"../../data_image/{folder}/{image}"))
            if index_image == (nb_images - 2):
                #to assign the image to the corresponding position in y_pred
                y_pred[index_folder, :, :] = image_array

    # Splitting y_pred into y_train_pred and y_test_pred
    total_samples = len(folder_list)
    train_samples = int(total_samples * train_size)
    y_train_pred_baseline = y_pred[:train_samples, :, :]
    y_test_pred_baseline = y_pred[train_samples:, :, :]

    return y_train_pred_baseline, y_test_pred_baseline


def score_baseline():
    X, y = create_and_preprocess_X_y()
    X_train, X_test, y_train, y_test = create_train_test_set(X,y,0.8)
    y_train_pred_baseline, y_test_pred_baseline = create_y_pred_baseline(0.8)
    epsilon = 1e-10  # Small constant to prevent division by zero

    numerator = abs(y_test - y_test_pred_baseline)
    denominator = y_test + y_test_pred_baseline + epsilon

    metric_baseline = numerator / denominator
    metric_baseline.sum()