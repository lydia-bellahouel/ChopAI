# Imports

import numpy as np
import os

from imageio import imwrite
from music21 import converter, instrument, note, chord

#################################

def extractNote(element):
    return int(element.pitch.ps)

#################################

def extractDuration(element):
    return element.duration.quarterLength

#################################

def get_notes(notes_to_parse):

    """
    Get all the notes and chords from the midi files into a dictionary containing:
        - Start: unit time at which the note starts playing
        - Pitch: pitch of the note
        - Duration: number of time units the note is played for
    """
    durations = []
    notes = []
    start = []

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            if element.isRest:
                continue

            start.append(element.offset)
            notes.append(extractNote(element))
            durations.append(extractDuration(element))

        elif isinstance(element, chord.Chord):
            if element.isRest:
                continue
            for chord_note in element:
                start.append(element.offset)
                durations.append(extractDuration(element))
                notes.append(extractNote(chord_note))

    return {"start":start, "pitch":notes, "dur":durations}

#################################

def midi2image(midi_path, max_repetitions = float("inf"), resolution = 0.25, lowerBoundNote = 21, upperBoundNote = 127, maxSongLength = 100):

    """
    1) Transform a midi file into a set of images:
        - Each image has a size of 106 (all notes between lowerBound and upperBound) x 100 time units (maxSongLength)
        - One time unit corresponds to 0.25 (resolution) beat from the original music
    2) Store images into the corresponding sub-folder (identified by music piece name) of the 'data_image' folder
    """

    output_folder = f"data_image/{midi_path.split('/')[-1].replace('.mid', '')}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    mid = converter.parse(midi_path)

    instruments = instrument.partitionByInstrument(mid)

    data = {}

    try:
        i=0
        for instrument_i in instruments.parts:
            notes_to_parse = instrument_i.recurse()

            notes_data = get_notes(notes_to_parse)
            if len(notes_data["start"]) == 0:
                continue

            if instrument_i.partName is None:
                data["instrument_{}".format(i)] = notes_data
                i+=1
            else:
                data[instrument_i.partName] = notes_data

    except:
        notes_to_parse = mid.flat.notes
        data["instrument_0"] = get_notes(notes_to_parse)

    for instrument_name, values in data.items():

        pitches = values["pitch"]
        durs = values["dur"]
        starts = values["start"]

        index = 0
        while index < max_repetitions:
            matrix = np.zeros((upperBoundNote-lowerBoundNote,maxSongLength))


            for dur, start, pitch in zip(durs, starts, pitches):
                dur = int(dur/resolution)
                start = int(start/resolution)

                if not start > index*(maxSongLength+1) or not dur+start < index*maxSongLength:
                    for j in range(start,start+dur):
                        if j - index*maxSongLength >= 0 and j - index*maxSongLength < maxSongLength:
                            matrix[pitch-lowerBoundNote,j - index*maxSongLength] = 255

            if matrix.any(): # If matrix contains no notes (only zeros) don't save it
                output_filename = os.path.join(output_folder, midi_path.split('/')[-1].replace(".mid",f"_{instrument_name}_{index}.png"))
                imwrite(output_filename,matrix.astype(np.uint8))
                index += 1
            else:
                break

#################################

def get_midi_data_as_images():

    """
    Iterate on all midi files from the 'data_raw' folder to:
        - Keep music pieces with one piano only
        - Store all corresponding images into 'data_image' file
    """
    # Storing all midi files into a 'files_raw' list
    files_raw = [file for file in os.listdir('data_raw')]

    # Storing all midi files with only one piano in a 'files' list
    files = []
    for file in files_raw:
        try:
            mid = converter.parse(f'data_raw/{file}')
            file_instruments = instrument.partitionByInstrument(mid)
            if len(file_instruments)==1:
                files.append(file)
        except:
            pass

    # Iterating on all files from 'files' list to create images
    for file in files:
        file_path = f"data_raw/{file}"
        midi2image(file_path)

#################################


if __name__ == '__main__':
    try:
        get_midi_data_as_images()
        print("✅ images created")
    except:
        print("❌ data transformation could not run")
