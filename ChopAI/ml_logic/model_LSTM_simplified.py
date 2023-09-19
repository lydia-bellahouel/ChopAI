# Misc
import pickle

# Data manipulation
import numpy as np
import pandas as pd

# Music
import music21 as m21
from music21 import converter, instrument, note, chord, stream

# Data Visualiation
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

# System
import os
import random
import shutil

# Performance metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping



#################################

def initialize_and_compile_LSTM_model(X_train, n_vocab_train):

    """Define model architecture"""

    """Define model architecture"""
    model = Sequential()
    model.add(LSTM(
        int(n_vocab_train),
        input_shape=(X_train.shape[1], X_train.shape[2]),
        return_sequences=True,
    ))

    model.add(LSTM(int(n_vocab_train/2)))

    model.add(Dense(n_vocab_train))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    print('Network is created')

    print("✅Model initialized and compiled")

    return model

#################################

def train_model(model, X_train, y_train):

    #checkpoints & callbacks:

    os.makedirs("../../checkpoint_lstm", exist_ok=True)
    file_path = "../../checkpoint_lstm/best_weights.h5"

    checkpoint = ModelCheckpoint(
        file_path,
        monitor='accuracy',
        verbose=0,
        save_best_only=True,
        mode='max'
    )


    callbacks_list = [checkpoint]

    # train model :

    history = model.fit(X_train, y_train, epochs=150, batch_size=64,callbacks=callbacks_list, validation_split=0.2, verbose=1)

    print("✅Model trained")

    return history
