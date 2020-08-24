import config
import utils
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding, TimeDistributed,Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint


class MusicGeneratorModel:
    def __init__(self, n_layers,retrain = False):
        if not retrain:
            self.model = Sequential()
            self.model.add(Embedding(n_layers, config.EMBEDDING_DIM,
                                    input_length=config.TIMESTAMP))
            self.model.add(Bidirectional(LSTM(256, return_sequences=True)))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.25))
            self.model.add(Bidirectional(LSTM(256)))
            self.model.add(BatchNormalization())
            self.model.add(Dense(256, activation="relu"))
            self.model.add(Dropout(0.25))
            self.model.add(Dense(n_layers, activation="softmax"))
            self.model.compile(
                loss=config.LOSS, optimizer=config.OPTIMIZER, metrics=['accuracy'])
        else:
            print("Retraining...")
            self.model = utils.load()

    def train(self, X_train, y_train, X_test, y_test):
        mc = ModelCheckpoint(config.MODEL_PATH, monitor='val_accuracy',
                             mode='max', save_best_only=True, verbose=1)
        history = self.model.fit(X_train, y_train, validation_data=(
            X_test, y_test), epochs=config.EPOCHS, callbacks=[mc])
        utils.save(self.model,config.FINAL_MODEL)
        return history
