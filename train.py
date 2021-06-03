import data_processing as dp
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras import models
import math
from keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

if not os.path.exists(f'{dp.train_file_path_txt}') or not os.path.exists(
        f'{dp.test_file_path_txt}') or not os.path.exists(f'{dp.validation_dataset_path_json}'):
    dp.split_and_write_dataset_paths()

if not os.path.exists(f'{dp.train_dataset_path_json}') or not os.path.exists(
        f'{dp.test_dataset_path_json}') or not os.path.exists(f'{dp.validation_dataset_path_json}'):
    dp.write_datasets_to_json()

model_feature: dp.Feature = dp.Feature.Spectrogram  # choose whether to train model with Cepstrum features or Spectrograms
model_name = "Spectrogram-imbalanced" if model_feature == dp.Feature.Spectrogram else "Cepstrum-imbalanced"
feature_key = "spectrogram" if model_feature == dp.Feature.Spectrogram else "cepstrum_feature"


# Function for shaping labels to contain value
def shape_labels(data_labels, all_labels):
    label_dict = {}
    for i, label in enumerate(all_labels):
        label_dict[label] = i

    result = pd.Series(data_labels)
    y = result.map(label_dict)
    return y


def main():
    train = dp.load_dataset_from_json(dp.train_dataset_path_json, model_feature)
    test = dp.load_dataset_from_json(dp.test_dataset_path_json, model_feature)

    shape_x = []
    shape_y = []
    labels = []

    # read labels from file
    label_file = open(dp.labels_path_txt)
    labels = label_file.readlines()
    label_file.close()

    # Getting shapes from train data
    for data in train:
        shape_x.append(len(data[feature_key]))
        shape_y.append(len(data[feature_key][0]))

    print(f'ROWS min:{np.min(shape_x)} max:{np.max(shape_x)} average:{np.average(shape_x)}')
    print(f'COLUMN min:{np.min(shape_y)} max:{np.max(shape_y)} average:{np.average(shape_y)}')

    # Calculating shape from train data
    input_shape = (math.floor(np.average(shape_x)), math.floor(np.average(shape_y)), 1)

    # Inspired by: https://www.tensorflow.org/tutorials/audio/simple_audio#evaluate_test_set_performance
    model = models.Sequential([
        layers.Conv2D(filters=34,
                      kernel_size=(3, 3),
                      activation='relu',
                      input_shape=input_shape,
                      padding='same'),
        layers.Conv2D(filters=64,
                      kernel_size=(3, 3),
                      activation='relu',
                      padding='same'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.35),
        layers.Dense(248, activation='relu'),
        layers.Dropout(0.1),
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
        features = data[feature_key]
        features = np.array(features)
        train_data.append(features)
        train_labels.append(data["label"])

    for data in test:
        features = data[feature_key]
        features = np.array(features)
        test_data.append(features)
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

    # set early stopping criteria
    pat = 2  # this is the number of epochs with no improvment after which the training will stop
    early_stopping = EarlyStopping(monitor='val_loss', patience=pat, verbose=1)

    # define the model checkpoint callback -> this will keep on saving the model as a physical file
    model_checkpoint = ModelCheckpoint(f'{os.path.join(dp.trained_model_path, model_name)}.h5', verbose=1,
                                       save_best_only=True)

    # Training model
    epochs = 15
    history = model.fit(train_data,
                        train_labels,
                        batch_size=80,
                        validation_data=(test_data, test_labels),
                        epochs=epochs,
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)

    # Evaluate model
    print("Val Score: ", model.evaluate(test_data, test_labels))

    # Plot model training history (loss, val_loss)
    title = f"{model_name} - Training History (Loss)"
    metrics = history.history
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.title(title)
    plt.legend(['Loss', 'Validation Loss'])
    plt.savefig(os.path.join(dp.image_path, title))
    plt.show()

    # Plot model training history (acc, val_acc)
    plt.figure()
    title = f"{model_name} - Training History (Accuracy)"
    plt.plot(history.epoch, metrics['acc'], metrics['val_acc'])
    plt.title(title)
    plt.legend(['Accuracy', 'Validation Accuracy'])
    plt.savefig(os.path.join(dp.image_path, title))
    plt.show()

    pass


# main()
