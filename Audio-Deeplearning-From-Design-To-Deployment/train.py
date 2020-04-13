import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import librosa
import os
import json
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description = 'training options')

parser.add_argument('--data_source', help = "'from_fold: from folder, json: from json file", type = str, default='from_fold')
args = vars(parser.parse_args())

DATA_PATH = "data.json"
SAVED_MODEL_PATH = "model.h5"
EPOCHS = 40
BATCH_SIZE = 32
PATIENCE = 5
LEARNING_RATE = 0.0001



def preprocess_dataset(dataset_path, SAMPLES_TO_CONSIDER: int, num_mfcc = 13, n_fft = 2048, hop_length = 512):
    """Extracts MFCCs from music dataset and saves them into a json file.

    :param dataset_path (str): Path to dataset
    :param json_path (str): Path to json file used to save MFCCs
    :param num_mfcc ( int): Number of coefficients to extract
    :param n_nfft (int): Interval we consider to apply FFT. Measures in # of samples
    :param hop_length (int): Sliding window for FFT. Measures in # of samples
    :return:
    """

    data = {
        'mapping': [],
        'labels': [],
        'MFCCs': [],
        'files': []
    }

    # loop through all sub-dirs
    total_samples = 0
    valid_samples = 0
    for i, (dirpath, dirname, filenames) in tqdm(enumerate(os.walk(dataset_path))):

        # ensure we're at sub-folder level
        if dirpath is not dataset_path:
            # save label (i.e., sub-folder name) in the mapping
            label = dirpath.partition('speech_commands_subset')[-1][1:]

            data['mapping'].append(label)
            print("\nProcessing: '{}'".format(label))
            print("number of files for each class: ", len(filenames))
            # process all audio files
            for f in filenames:
                total_samples += 1
                file_path = os.path.join(dirpath, f)

                # load audio file and slice it to ensure length consistency among different files
                signal, sample_rate = librosa.load(file_path)
                # print(signal.shape)
                # print(type(signal[0]))

                # drop audio files with less than pre-decided number of samples
                if len(signal) >= SAMPLES_TO_CONSIDER:
                    valid_samples += 1
                    # ensure consistency of the length of the signal
                    signal = signal[:SAMPLES_TO_CONSIDER]

                    # extract MFCCs
                    MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc = num_mfcc, n_fft = n_fft, 
                        hop_length = hop_length) 
                    # print(MFCCs.shape)
                    # print(type(MFCCs[0,0]))

                    # store data for analysed track
                    data['MFCCs'].append(MFCCs.T.tolist())
                    data['labels'].append(i-1)
                    # data['files'].append(file_path)
                    # print("{}: {}".format(file_path, i-1))

                    # if valid_samples == 20:
                    #     valid_samples =0
                    #     break
    print("\ntotal samples: ", total_samples)
    print("\nvalid_samples: ", valid_samples)

    
    return data


def load_data_from_fold(data_path):
    """Loads training dataset from json file.
    :param data_path (str): Path to the folder containing dataset
    :return X (ndarray): Inputs
    :return y (ndarray): Targets
    """
    print("\nLoading data from json folder {}".format(data_path))

    SAMPLES_TO_CONSIDER = 22050

    data = preprocess_dataset(data_path, SAMPLES_TO_CONSIDER)

    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    print("Training sets loaded!")
    print("data size :", X.shape, "labels size: ", y.shape)
    print("release the 'data' for memories")
    del data

    return X, y

def load_data_from_json(json_path):
    """Loads training dataset from json file.
    :param data_path (str): Path to json file containing data
    :return X (ndarray): Inputs
    :return y (ndarray): Targets
    """
    print("\nLoading data from json file")
    with open(json_path, "r") as fp:
        data = json.load(fp)
    
    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    print("Training sets loaded!")
    print("data size :", X.shape, "labels size: ", y.shape)
    print("release the 'data' for memories")
    del data

    return X, y


def prepare_dataset(data_path, test_size=0.2, validation_size=0.2):
    """Creates train, validation and test sets.
    :param data_path (str): Path to json file containing data
    :param test_size (flaot): Percentage of dataset used for testing
    :param validation_size (float): Percentage of train set used for cross-validation
    :return X_train (ndarray): Inputs for the train set
    :return y_train (ndarray): Targets for the train set
    :return X_validation (ndarray): Inputs for the validation set
    :return y_validation (ndarray): Targets for the validation set
    :return X_test (ndarray): Inputs for the test set
    :return X_test (ndarray): Targets for the test set
    """

    # load dataset
    if data_path.endswith('json'):
        X, y = load_data_from_json(data_path)
    else:
        X, y = load_data_from_fold(data_path)
    # create train, validation, test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # add an axis to nd array
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]

    return X_train, y_train, X_validation, y_validation, X_test, y_test


def build_model(input_shape, loss="sparse_categorical_crossentropy", learning_rate=0.0001):
    """Build neural network using keras.
    :param input_shape (tuple): Shape of array representing a sample train. E.g.: (44, 13, 1)
    :param loss (str): Loss function to use
    :param learning_rate (float):
    :return model: TensorFlow model
    """

    # build network architecture using convolutional layers
    model = tf.keras.models.Sequential()

    # 1st conv layer
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape,
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'))

    # 2nd conv layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'))

    # 3rd conv layer
    model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2,2), padding='same'))

    # flatten output and feed into dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    tf.keras.layers.Dropout(0.3)

    # softmax output layer
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    optimiser = tf.optimizers.Adam(learning_rate=learning_rate)

    # compile model
    model.compile(optimizer=optimiser,
                  loss=loss,
                  metrics=["accuracy"])

    # print model parameters on console
    model.summary()

    return model


def train(model, epochs, batch_size, patience, X_train, y_train, X_validation, y_validation):
    """Trains model
    :param epochs (int): Num training epochs
    :param batch_size (int): Samples per batch
    :param patience (int): Num epochs to wait before early stop, if there isn't an improvement on accuracy
    :param X_train (ndarray): Inputs for the train set
    :param y_train (ndarray): Targets for the train set
    :param X_validation (ndarray): Inputs for the validation set
    :param y_validation (ndarray): Targets for the validation set
    :return history: Training history
    """

    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="accuracy", min_delta=0.001, patience=patience)

    # train model
    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_validation, y_validation),
                        callbacks=[earlystop_callback])
    return history


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
    :param history: Training history of model
    :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="accuracy")
    axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")

    # create loss subplot
    axs[1].plot(history.history["loss"], label="loss")
    axs[1].plot(history.history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation")

    plt.show()




if __name__ == "__main__":
    DATASET_PATH = 'D:/python/Data/Audio/speech_commands_subset'
    JSON_PATH = 'data.json'
    # generate train, validation and test sets
    if args['data_source'] == 'json':
        X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_dataset(JSON_PATH)
    else:
        X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_dataset(DATASET_PATH)

    # create network
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape, learning_rate=LEARNING_RATE)

    # train network
    history = train(model, EPOCHS, BATCH_SIZE, PATIENCE, X_train, y_train, X_validation, y_validation)

    # plot accuracy/loss for training/validation set as a function of the epochs
    plot_history(history)

    # evaluate network on test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("\nTest loss: {}, test accuracy: {}".format(test_loss, 100*test_acc))

    # save model
    model.save(SAVED_MODEL_PATH)