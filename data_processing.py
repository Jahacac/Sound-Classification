import numpy as np
import os
from snd_cut import cut_path
import librosa
import re
import librosa.display
import json
from enum import Enum
import matplotlib.pyplot as plt
from sklearn.utils import resample


# Feature enumerator
class Feature(Enum):
    Cepstrum = 1
    Spectrogram = 2


# Data distribution enumerator
class DatasetDistribution(Enum):
    Balanced = 1
    Imbalanced = 2


# files that contain dataset folder paths
train_file_path_txt = os.path.join('data', 'train_set.txt')
test_file_path_txt = os.path.join('data', 'test_set.txt')
validation_file_path_txt = os.path.join('data', 'validation_set.txt')

# txt file that contains all labels
labels_path_txt = os.path.join('data', 'labels.txt')

# files that contain dataset values
train_dataset_path_json = os.path.join('data', 'train_dataset.json')
test_dataset_path_json = os.path.join('data', 'test_dataset.json')
validation_dataset_path_json = os.path.join('data', 'validation_dataset.json')

image_path = os.path.join('images')
trained_model_path = os.path.join('trained_models')

# For replication purposes
RANDOM_SEED = 1
np.random.seed(RANDOM_SEED)


# split data into train, test, validation datasets
def split_data():
    train = []
    test = []
    validation = []
    for sound_folder in os.scandir(f'{cut_path}'):
        sound_data = []
        for sound_file in os.scandir(sound_folder):
            sound_data.append(
                os.path.join(cut_path, os.path.basename(sound_folder.name), os.path.basename(sound_file.name)))
        np.random.shuffle(sound_data)  # shuffle data

        # determine train and test ending index in data
        train_index_end = int(len(sound_data) * 0.8)  # 0.8
        test_index_end = int(len(sound_data) * 0.8) + int(len(sound_data) * 0.1)  # 0.8, 0.1

        # get train and test from data
        train = train + sound_data[:train_index_end]
        test = test + sound_data[train_index_end:test_index_end]
        validation = validation + sound_data[test_index_end:]

    return train, test, validation


# write train, test, validation datasets to txt files
def write_split_data(train, test, validation):
    train_file = open(train_file_path_txt, "w")
    train_file.write("\n".join(train))
    train_file.close()

    test_file = open(test_file_path_txt, "w")
    test_file.write("\n".join(test))
    test_file.close()

    validation_file = open(validation_file_path_txt, "w")
    validation_file.write("\n".join(validation))
    validation_file.close()
    return


# extract train, test, validation dataset paths and write to file
def split_and_write_dataset_paths():
    train_dataset, test_dataset, validation_dataset = split_data()
    write_split_data(train_dataset, test_dataset, validation_dataset)


# get sound file names within dataset paths
def get_dataset_sound_filenames(dataset_path):
    sound_paths = []
    dataset = open(dataset_path)

    # get sound .wav files from dataset
    for i, sound_file_path in enumerate(dataset):
        sound_file_path = sound_file_path.replace('\n', '')
        sound_paths.append(sound_file_path)
    dataset.close()
    return sound_paths


# extract label from sound path
def get_label_from_sound_path(sound_path):
    label = re.search(r'_(.*?).wav', sound_path).group(1)
    return label


# get dataset from dataset_path, returns List(dict(song_name, cepstrum_features, label))
def get_dataset(dataset_path):
    result = []
    sound_paths = get_dataset_sound_filenames(dataset_path)
    labels = [get_label_from_sound_path(sound_path) for sound_path in sound_paths]
    print('started extracting features!')
    cepstrum_features = [get_feature(sound_path, Feature.Cepstrum, i == 0) for i, sound_path in
                         enumerate(sound_paths)]  # Plot only the first feature
    print('finished extracting cepstrum features!')
    spectrograms = [get_feature(sound_path, Feature.Spectrogram, i == 0) for i, sound_path in enumerate(sound_paths)]
    print('finished extracting spectrograms!')

    for sound_path, label, cepstrum_feature, spectrogram in zip(sound_paths, labels, cepstrum_features, spectrograms):
        result.append({
            "sound_path": sound_path,
            "label": label,
            "cepstrum_feature": cepstrum_feature.tolist(),
            "spectrogram": spectrogram.tolist()
        })
    return result


def get_feature(sound_path: str, feature_enum: Feature, plot_result: bool):
    y, sr = librosa.load(sound_path)
    result = None
    title = ""

    if feature_enum == Feature.Cepstrum:
        result = librosa.feature.mfcc(y=y, sr=sr, n_fft=512)
        title = "Cepstrum Features"
    else:
        result = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512)
        result = librosa.power_to_db(result, ref=np.max)
        title = "Spectrogram"

    if plot_result:
        fig, ax = plt.subplots()
        img = librosa.display.specshow(result, x_axis='time', ax=ax)
        fig.colorbar(img, ax=ax)
        ax.set(title=title)
        plt.savefig(os.path.join(image_path, title))
        plt.show()

    return result


# get train/test/validation dataset as json (format: sound_path, label, cepstrum_features)
# write to json file
def write_datasets_to_json():
    train_data = get_dataset(train_file_path_txt)
    train = json.dumps(train_data)
    train_dataset_file = open(train_dataset_path_json, 'w')
    train_dataset_file.write(train)
    train_dataset_file.close()
    print('train_dataset_file finished!')

    test_data = get_dataset(test_file_path_txt)
    test = json.dumps(test_data)
    test_dataset_file = open(test_dataset_path_json, 'w')
    test_dataset_file.write(test)
    test_dataset_file.close()
    print('test_dataset_file finished!')

    validation_data = get_dataset(validation_file_path_txt)
    validation = json.dumps(validation_data)
    validation_dataset_file = open(validation_dataset_path_json, 'w')
    validation_dataset_file.write(validation)
    validation_dataset_file.close()
    print('validation_dataset_file finished!')

    # extract all labels and write distinct values
    labels = []
    for data in train_data:
        labels.append(data["label"])
    for data in test_data:
        labels.append(data["label"])
    for data in validation_data:
        labels.append(data["label"])

    labels_file = open(labels_path_txt, 'w')
    labels = sorted(list(set(labels)))  # select distinct labels
    labels_file.writelines("\n".join(labels))
    labels_file.close()
    print('labels_file finished!')


# load dataset and return dataset values
def load_dataset_from_json(dataset_path, model_feature: Feature):
    file = open(dataset_path)
    json_data = json.loads(file.readline())
    file.close()
    # determining which feature to delete from data
    delete_feature_key = "cepstrum_feature" if model_feature == Feature.Spectrogram else "spectrogram"
    for data in json_data:
        if delete_feature_key in data:
            del data[delete_feature_key]
    return json_data


# get distinct labels
def get_labels():
    label_file = open(labels_path_txt)
    labels = label_file.read()
    labels = labels.split('\n')
    label_file.close()
    return labels


# plot label distribution in bar plot
def plot_label_distribution(data, histogram_name: str):
    labels = get_labels()
    label_freq = {}

    for label in labels:
        label_freq[label] = 0

    for item in data:
        label_freq[item["label"]] += 1

    plt.figure(figsize=(10, 5))
    plt.bar(label_freq.keys(), label_freq.values(), color='g')
    plt.title(histogram_name)
    plt.savefig(os.path.join(image_path, histogram_name))
    plt.show()


# Balances dataset so that all labels have the same number of data
# returns balanced dataset
def balance_dataset(dataset):
    n_samples = 1000
    labels = get_labels()
    retval = []

    for label in labels:
        dataset_for_label = []  # all data within dataset that has this label
        for data in dataset:
            if data["label"] == label:
                dataset_for_label.append(data)

        if len(dataset_for_label) < n_samples:
            dataset_for_label = resample(dataset_for_label,
                                         replace=True,
                                         n_samples=n_samples,  # len(no_claim),
                                         random_state=RANDOM_SEED)
        retval = retval + dataset_for_label
    return retval

# split_and_write_dataset_paths()
# write_datasets_to_json()
