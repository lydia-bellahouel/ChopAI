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
import tensorflow.keras.regularizers as regularizers

#################################
#################################


def predict_sequence(X_train, X_test, n_vocab_train, pitchnames_train):
    """Generate a predicted sequence of notes following X_test notes"""

    # load best weights
    path = "../../checkpoint_lstm/best_weights.h5"
    model = initialize_and_compile_LSTM_model(X_train, n_vocab_train)
    model.load_weights(path)

    # pick a random sequence from the input as a starting point for the prediction
    start = np.random.randint(0, len(X_test)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames_train))

    pattern = X_test[start]
    y_pred = []

    id_notes = np.arange(0, n_vocab_train)
    # generate 500 notes
    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab_train)

        prediction_probabilities = model.predict(prediction_input, verbose=0)

        index = np.random.choice(id_notes,1, p = prediction_probabilities[0])
        #print(index)
        result = int_to_note[index[0]]
        y_pred.append(result)

        pattern.append(index[0])
        pattern = pattern[1:len(pattern)]

    return y_pred, pattern

#################################

def notes_to_midi_files(y_pred):

    """Create midi files from notes"""

    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for element in y_pred:
        # pattern is a chord
        if ('.' in element) or element.isdigit():
            notes_in_chord = element.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(element)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    os.makedirs("../../generated_music_from_lstm", exist_ok=True)

    midi_stream.write('midi', fp='../../generated_music_from_lstm')
