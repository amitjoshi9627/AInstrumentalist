import dataset
import utils
from model import MusicGeneratorModel


def train():
    data, file_ = dataset.get_data()
    X_train, X_test, y_train, y_test = data
    utils.save_file(file_)
    file_path = "../saved_model/X_val.pickel"
    utils.save_file(X_test, file_path)
    num_layers = len(file_['X1'])

    model = MusicGeneratorModel(num_layers,retrain=False)
    history = model.train(X_train, y_train, X_test, y_test)
    utils.plot_graphs(history)


if __name__ == "__main__":
    train()
