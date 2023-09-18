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
#import tensorflow.keras.regularizers as regularizers


#################################

def initialize_and_compile_LSTM_model(X_train, n_vocab_train):

    """Define model architecture"""

    model = Sequential()
    model.add(LSTM(
        int(n_vocab_train),
        input_shape=(X_train.shape[1], X_train.shape[2]),
        return_sequences=True,
    ))


    model.add(LSTM(
            n_vocab_train,
            return_sequences=True,
            recurrent_dropout=0.3,
        ))

    model.add(LSTM(int(n_vocab_train/2)))


    model.add(BatchNorm())
    model.add(Dropout(0.3))


    model.add(Dense(n_vocab_train))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

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

    es = EarlyStopping(patience=5, restore_best_weights=True)

    callbacks_list = [checkpoint,es]


    # train model :

    history = model.fit(X_train, y_train, epochs=200, batch_size=32 ,callbacks=callbacks_list, validation_split=0.2, verbose=1)

    return history
