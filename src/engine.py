import config
import utils
import numpy as np
import os
from music21 import chord, note, instrument, stream


def _convert_to_midi(predicted_notes):
    output_notes = []
    offset_ = 0

    for pattern in predicted_notes:

        if "." in pattern or pattern.isdigit():

            chord_list = pattern.split(".")
            notes = []
            for note_ in chord_list:
                new_note = note.Note(int(note_))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)

            new_chord = chord.Chord(notes)
            new_chord.offset = offset_
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset_
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset_ += 0.5
    midi_stream = stream.Stream(output_notes)
    return _save_midi(midi_stream)


def _save_midi(midi_stream):
    def extractor(x):
        return x.split(".")[0].split("_")[-1]

    files = sorted(
        [
            int(extractor(f))
            for f in os.listdir(config.OUTPUT_MIDI)
            if f.endswith(".mid")
        ]
    )
    if files != []:
        index = files[-1] + 1
        print("INDEX", index, files)
    else:
        index = 0
    file_name = "Music_" + str(index) + ".mid"
    file_path = config.OUTPUT_MIDI + file_name
    return midi_stream.write(fmt="midi", fp=file_path)


def get_random_music():
    model = utils.load()
    X_val = utils.load_file(file_name=config.X_VAL_PATH)
    ind = np.random.randint(0, len(X_val) - 1)
    random_music = X_val[ind]
    predictions = []
    for i in range(config.PRED_ITERATIONS):
        random_music = random_music.reshape(1, config.TIMESTAMP)
        prediction = model.predict_classes(random_music)
        predictions.append(prediction[0])
        random_music = np.insert(random_music[0], len(random_music[0]), prediction[0])
        random_music = random_music[1:]

    file_ = utils.load_file()
    X_int_to_note = file_["X2"]

    predicted_notes = [X_int_to_note[note_] for note_ in predictions]
    return _convert_to_midi(predicted_notes)
