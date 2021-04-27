import numpy as np
import os
from snd_cut import cut_path
import librosa
import re
import librosa.display
import matplotlib.pyplot as plt

train_file_path = os.path.join('data', 'train_set.txt')
test_file_path = os.path.join('data', 'test_set.txt')
validation_file_path = os.path.join('data', 'validation_set.txt')

# For replication purposes
np.random.seed(1)

# split data into train, test, validation datasets
def split_data():
    data = []
    for wav_folder in os.scandir(f'{cut_path}'):
        data.append(os.path.join(cut_path, os.path.basename(wav_folder.name)))
    np.random.shuffle(data)  # shuffle data
    # determine train and test ending index in data
    train_index_end = int(len(data) * 0.8)
    test_index_end = int(len(data) * 0.8) + int(len(data) * 0.1)
    # get train and test from data
    train = data[:train_index_end]
    test = data[train_index_end:test_index_end]
    validation = data[test_index_end:]
    return train, test, validation


# write train, test, validation datasets to txt files
def write_split_data(train, test, validation):
    train_file = open(train_file_path, "w")
    train_file.write("\n".join(train))
    train_file.close()

    test_file = open(test_file_path, "w")
    test_file.write("\n".join(test))
    test_file.close()

    validation_file = open(validation_file_path, "w")
    validation_file.write("\n".join(validation))
    validation_file.close()
    return


# extract train, test, validation dataset and write to file
def split_and_write_dataset():
    train_dataset, test_dataset, validation_dataset = split_data()
    write_split_data(train_dataset, test_dataset, validation_dataset)


# get sound file names within dataset
def get_dataset_sound_filenames(dataset_path):
    sound_paths = []
    dataset = open(dataset_path)

    # get .wav folders from dataset
    for i, wav_folder_path in enumerate(dataset):
        wav_folder_path = wav_folder_path.replace('\n', '')
        # get sounds from all .wav folders
        for sound_path in os.scandir(os.path.join(wav_folder_path)):
            sound_paths.append(os.path.join(wav_folder_path, sound_path.name))
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
    cepstrum_features = [get_cepstrum_features(sound_path) for sound_path in sound_paths]

    for sound_path, label, cepstrum_feature in zip(sound_paths, labels, cepstrum_features):
        result.append({
            "sound_path": sound_path,
            "label": label,
            "cepstrum_feature": cepstrum_feature
        })
    return result


# get cepstrum features from file at file_path
def get_cepstrum_features(sound_path):
    y, sr = librosa.load(sound_path)
    result = librosa.feature.mfcc(y=y, sr=sr, n_fft=512)
    # fig, ax = plt.subplots()
    # img = librosa.display.specshow(result, x_axis='time', ax=ax)
    # fig.colorbar(img, ax=ax)
    # ax.set(title='MFCC')
    # plt.show()
    return result


# paths = get_dataset_sound_filenames(validation_file_path)
# label = attach_labels_to_sound_paths(paths)[0][0]
# get_cepstrum_features(label)

# path na dataset_folder
# dohvatit sve njegove splittane glasove ->


# dohvatit train,test,validation .wav imena datoteka
# dohvatit njihove cuts (.wav datoteka, label = sound)
# .wav cuts pretvorit u kepstralne znacajke, resize na average shape
