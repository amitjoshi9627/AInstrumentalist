from tensorflow.keras.losses import categorical_crossentropy

LOSS = categorical_crossentropy
OPTIMIZER = "adam"
INPUT_SHAPE = None

TARGET_SIZE = None
BATCH_SIZE = 32
EPOCHS = 150

STEPS_PER_EPOCH = 8
VALIDATION_STEPS = 2

MODEL_SAVE_PATH = "../saved_model/"

DATA_PATH = "../data/midi/"
MODEL_PATH = "../saved_model/Model.h5"
FILE_NAME = "../saved_model/file.pickel"
OUTPUT_MIDI = "../saved_music/"
NUMPY_SAVEPATH = "../saved_model/numpy.npy"
FINAL_MODEL = "../saved_model/FinalModel.h5"
X_VAL_PATH = "../saved_model/X_val.pickel"

INSTRUMENT = "Piano"
FREQ_THRESH = 2
TIMESTAMP = 48
EMBEDDING_DIM = 128

PRED_ITERATIONS = 120
