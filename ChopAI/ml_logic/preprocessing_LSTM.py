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
from tensorflow.keras.utils import to_categorical

###################

def load_midi_into_notes():
    """Convert train files into notes"""

    # Extracted notes for each file
    notes_train = []

    # Save the indexes of notes in which a new composition started to create consistent sequences
    new_composition_indexes_train = []

    # Load all train MIDI files into notes
    input_path_train = "../../data_train"
    for file in os.listdir(input_path_train):
        file_path = os.path.join(input_path_train, file)

        # Add new index to alert that it's a new composition
        if new_composition_indexes_train and new_composition_indexes_train[-1] != len(notes_train):
            new_composition_indexes_train.append(len(notes_train))

        try:
            # Convert the music into a score object
            score = converter.parse(file_path)

            print("Parsing %s" % file)

            elements_in_part = None

            try:  # File has instrument parts
                # Given a score that represents the MIDI, partition it into parts for each unique instrument found
                partitions_by_instrument = instrument.partitionByInstrument(score)
                # Visit all the elements (notes, chords, rests, and more) of each of its internal "measures."
                elements_in_part = partitions_by_instrument.parts[0].recurse()

            except:  # File has notes in a flat structure
                elements_in_part = score.flat.notes

            # Scroll through all the elements (notes or chords) picked up
            for element in elements_in_part:
                # If the element is a note...
                if isinstance(element, note.Note):
                    # Add note to array
                    notes_train.append(str(element.pitch))
                # If the element is a chord (a set of notes --> e.g., C4 F4)
                elif isinstance(element, chord.Chord):
                    # Extract each note from the chord and insert it into the array in the format Note1.Note2.Note3
                    notes_train.append('.'.join(str(n) for n in element.normalOrder))

        except Exception as e:
            print(f"Error parsing {file}: {str(e)}")
            continue


    print("✅Loading notes done for train files")

    # Save the 'notes_train' and 'notes_test' list to a pickle file
    os.makedirs("../../data_vocab/",exist_ok=True)
    with open("../../data_vocab/notes_train.pkl", "wb") as filepath:
        pickle.dump(notes_train, filepath)

    # Return notes and new composition indexes
    return notes_train, new_composition_indexes_train


#################################

def create_vocabulary(notes_train):

    """define notes vocabulary based on train set"""

    # get all pitch names
    pitchnames_train = sorted(set(item for item in notes_train))

    # create a dictionary to map pitches to integers
    note_to_int_train = dict((note, number) for number, note in enumerate(pitchnames_train))

    n_vocab_train = len(set(notes_train))

    print("✅Vocabulary created")

    return note_to_int_train, n_vocab_train, pitchnames_train


#################################

def split_train_test(notes_train, new_composition_indexes_train, note_to_int_train,
                     n_vocab_train, pitchnames_train, sequence_length = 100):
    """
    Prepare the test and train sequences to be used by the Neural Network
    """

    # !!TRAIN!! split creation

    X_train = []
    y_train = []

    wait = 0
    # create input sequences and the corresponding outputs
    for i in range(0, len(notes_train) - sequence_length, 1):

        # if the ground truth index is a note/chord that belongs to a new composition
        if (i + sequence_length) in new_composition_indexes_train:
            wait = sequence_length - 1
            continue
        if wait != 0:
            wait = wait - 1
            continue

        sequence_in = notes_train[i:i + sequence_length]
        sequence_out = notes_train[i + sequence_length]

        X_train.append([note_to_int_train[char] for char in sequence_in])
        y_train.append(note_to_int_train[sequence_out])

    n_patterns = len(X_train)

    # reshape the input into a format compatible with LSTM layers
    X_train = np.reshape(X_train, (n_patterns, sequence_length, 1))

    # normalize input
    X_train = X_train / float(n_vocab_train)

    # one-hot encoding of the output
    y_train = to_categorical(y_train, num_classes=n_vocab_train)

    print("✅X_train and y_train created")

    return X_train, y_train
