from preprocessing_LSTM import create_train_test_folders, load_midi_into_notes, create_vocabulary, split_train_test
from model_LSTM_simplified import initialize_and_compile_LSTM_model, train_model

#GATHERING PREPROCESSING & MODEL FUNCTIONS TO GET LSTM MODEL

#Preprocessing data to get X_train, y_train, X_test, y_test:

create_train_test_folders()
notes_train, notes_test, new_composition_indexes_train, new_composition_indexes_test = load_midi_into_notes()
note_to_int_train, n_vocab_train, pitchnames_train = create_vocabulary(notes_train)
X_train, y_train, X_test, y_test = split_train_test(notes_train, notes_test, new_composition_indexes_train, new_composition_indexes_test, note_to_int_train, n_vocab_train, pitchnames_train, sequence_length = 100)

#generate and train LSTM model:
model = initialize_and_compile_LSTM_model(X_train, n_vocab_train)
history = train_model(model, X_train, y_train)
