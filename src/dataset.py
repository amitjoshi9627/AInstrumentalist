import config
import numpy as np
import os
import utils
from music21 import converter, instrument, note, chord
from collections import Counter
from tensorflow.keras.utils import to_categorical


def _print_message(message):
    print(f"[AInstrumentalist] (Data Processing) {message}")


def _get_notes_from_midi(file_name):

    file_path = os.path.join(config.DATA_PATH, file_name)
    midi = converter.parse(file_path)
    groups = instrument.partitionByInstrument(midi)
    notes = []
    piano_notes = [part for part in groups.parts if config.INSTRUMENT in str(part)]
    for part in piano_notes:
        note_to_parse = part.recurse()
        for element in note_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append(".".join([str(n_) for n_ in element.normalOrder]))
    if notes == []:
        return np.array([])
    return np.array(notes)


def _get_all_notes():
    if os.path.isfile(config.NUMPY_SAVEPATH):
        notes_list = np.load(config.NUMPY_SAVEPATH, allow_pickle=True)
    else:
        files = [f for f in os.listdir(config.DATA_PATH) if f.endswith(".mid")]
        message = f"Extracting notes from {len(files)} files."
        _print_message(message)
        notes = np.array([_get_notes_from_midi(f) for f in files], dtype="object")
        notes_list = []
        num_songs = 0
        for note_ in notes:
            if len(note_) != 0:
                num_songs += 1
                notes_list.append(note_)
        notes_list = np.array(notes_list, dtype="object")

        np.save(config.NUMPY_SAVEPATH, notes_list)
        print("No. of Songs for dataset:", num_songs)
    return notes_list


def _get_frequent_notes(note_list):
    note_list = np.array(note_list)
    notes = [ele for note_ in note_list for ele in note_]
    freq = Counter(notes)
    frequent_notes = [note for note, f in freq.items() if f >= config.FREQ_THRESH]
    new_notes = []
    for notes_ in note_list:
        temp = []
        for note_ in notes_:
            if note_ in frequent_notes:
                temp.append(note_)
        new_notes.append(temp)
    _print_message("Got Frequent Notes")
    return np.array(new_notes, dtype="object")


def _get_keys(arr):

    return dict([(i, j) for i, j in enumerate(arr)]), dict(
        [(j, i) for i, j in enumerate(arr)]
    )


def _get_timestamp_data(notes_data):
    X = []
    y = []
    for notes in notes_data:
        for i in range(0, len(notes) - config.TIMESTAMP):
            input_ = notes[i : i + config.TIMESTAMP]
            output_ = notes[i + config.TIMESTAMP]
            X.append(input_)
            y.append(output_)
    _print_message("Timestamp data created")
    return np.array(X), np.array(y)


def get_data():
    notes = _get_all_notes()
    notes_data = _get_frequent_notes(notes)

    X, y = _get_timestamp_data(notes_data)
    _print_message(f"Total Data points: {len(X)}")
    X_int_to_note, X_note_to_int = _get_keys(list(set(X.ravel())))
    y_int_to_note, y_note_to_int = _get_keys(list(set(y)))

    file_ = {
        "X1": X_note_to_int,
        "X2": X_int_to_note,
        "y1": y_note_to_int,
        "y2": y_int_to_note,
    }

    X_seq = []
    for i in X:
        temp = []
        for j in i:
            temp.append(X_note_to_int[j])
        X_seq.append(temp)
    X_seq = np.array(X_seq)
    y_seq = np.array([y_note_to_int[i] for i in y])
    y_seq = to_categorical(y_seq, num_classes=len(X_note_to_int))
    _print_message("Data Processing completed")

    return utils.train_test_split(X_seq, y_seq, test_size=0.2), file_
