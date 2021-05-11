import data_processing as dp
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras import models
import math
from keras.utils import to_categorical

if not os.path.exists(f'{dp.train_file_path_txt}') or not os.path.exists(f'{dp.test_file_path_txt}'):
    dp.split_and_write_dataset_paths()

if not os.path.exists(f'{dp.train_dataset_path_json}') or not os.path.exists(f'{dp.test_dataset_path_json}'):
    dp.write_datasets_to_json()


def extract_labels_for_model(data):
    # cepstrum features
    pass


# Extract cepstrum features and their shape
def extract_cepstrum_features_for_model(data):
    # cepstrum features
    pass


# Function for shaping labels to contain value
def shape_labels(data_labels, all_labels):
    label_dict = {}
    for i, label in enumerate(all_labels):
        label_dict[label] = i

    result = pd.Series(data_labels)
    y = result.map(label_dict)
    return y


def main():
    train = dp.load_dataset_from_json(dp.train_dataset_path_json)
    test = dp.load_dataset_from_json(dp.test_dataset_path_json)

    shape_x = []
    shape_y = []
    labels = []

    # Getting shapes and labels from train data
    for data in train:
        shape_x.append(len(data["cepstrum_feature"]))
        shape_y.append(len(data["cepstrum_feature"][0]))
        labels.append(data["label"])

    # Extracting all distinct labels
    labels = list(set(labels))

    print(f'ROWS min:{np.min(shape_x)} max:{np.max(shape_x)} average:{np.average(shape_x)}')
    print(f'COLUMN min:{np.min(shape_y)} max:{np.max(shape_y)} average:{np.average(shape_y)}')

    # Calculating shape from train data
    input_shape = (math.floor(np.average(shape_x)), math.floor(np.average(shape_y)), 1)

    # Inspired by: https://www.tensorflow.org/tutorials/audio/simple_audio#evaluate_test_set_performance
    model = models.Sequential([
        # preprocessing.Resizing(32, 32),
        # norm_layer,
        layers.Conv2D(filters=33,
                      kernel_size=(3, 3),
                      activation='relu',
                      input_shape=input_shape,
                      padding='same'),
        layers.Conv2D(filters=63,
                      kernel_size=(3, 3),
                      activation='relu',
                      padding='same'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(labels), activation='softmax')
    ])
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    # Preparing train/test data in format recognizable to model
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for data in train:
        cepstrum_features = data["cepstrum_feature"]
        cepstrum_features = np.array(cepstrum_features)
        train_data.append(cepstrum_features)
        train_labels.append(data["label"])

    for data in test:
        cepstrum_features = data["cepstrum_feature"]
        cepstrum_features = np.array(cepstrum_features)
        test_data.append(cepstrum_features)
        test_labels.append(data["label"])

    train_data = np.array(train_data)
    test_data = np.array(test_data)

    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], 1)
    test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], 1)

    # Shaping lables from string to numbers
    train_labels = shape_labels(train_labels, labels)
    test_labels = shape_labels(test_labels, labels)

    # Encoding labels
    train_labels = to_categorical(train_labels, len(labels))
    test_labels = to_categorical(test_labels, len(labels))

    # Training model
    EPOCHS = 10
    history = model.fit(train_data,
                        train_labels,
                        batch_size=50,
                        validation_data=(test_data, test_labels),
                        epochs=EPOCHS,
                        verbose=1)

    # Evaluate model
    print("Val Score: ", model.evaluate(test_data, test_labels))

    # Plot model training history (loss, val_loss)
    metrics = history.history
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.show()
    pass


main()
