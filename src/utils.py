import config
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from midi2audio import FluidSynth
from sklearn.model_selection import train_test_split as tts
from tensorflow.keras.models import load_model, save_model
from pydub import AudioSegment
from pydub.utils import which


def train_test_split(X, y, test_size=0.3, train_size=None, random_state=None):
    return tts(X, y, test_size=test_size)
    if random_state is not None:
        np.random.seed(seed=random_state)
    if train_size:
        test_size = 1.0 - train_size
    X = np.array(X)
    y = np.array(y)
    size = X.shape[0]
    indx = np.random.choice(size, int(size * test_size))

    return X[~indx], X[indx], y[~indx], y[indx]


def accuracy_score(data1, data2):
    return np.mean(np.array(data1) == np.array(data2))


def save(model):
    if not os.path.exists("../saved_model/"):
        os.makedirs('../saved_model/')
    model.save(config.MODEL_PATH)


def load():
    model = load_model(config.MODEL_PATH)
    return model


def save_file(file, file_name=None):
    if file_name is None:
        file_name = config.FILE_NAME
    with open(file_name, "wb") as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_file(file_name=None):
    if file_name is None:
        file_name = config.FILE_NAME
    with open(file_name, "rb") as handle:
        file = pickle.load(handle)
    return file


def plot_graphs(history):
    for string in ["accuracy", "loss"]:
        plt.plot(history.history[string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.show()


def midi_to_wav(src_file):
    dest_file = src_file.split(".")[0]+".wav"
    fs = FluidSynth()
    fs.midi_to_audio(src_file, dest_file)
    return dest_file


def convert_midi(src_file):
    AudioSegment.converter = which("ffmpeg")
    dest_file = src_file.split(".")[0]+".wav"
    sound = AudioSegment.from_file(src_file)
    sound.export(dest_file, format="wav")
    return dest_file
