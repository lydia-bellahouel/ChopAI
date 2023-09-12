# Imports
import os
import matplotlib.pyplot as plt
import numpy as np

#################################

def create_y_pred_baseline():
    """
    Based on model predicting last seen sequence, get:
        - y_train_pred: predictions for the train set
        - y_test_pred: predictions for the test set
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
    y_train_pred = y_pred[0:70, :, :]
    y_test_pred = y_pred[70:88, :, :]

    return y_train_pred, y_test_pred
