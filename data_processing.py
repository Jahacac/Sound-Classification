import numpy as np
import os
from snd_cut import cut_path
import librosa
import re
import librosa.display
import json
import matplotlib.pyplot as plt

# files that contain dataset folder paths
train_file_path_txt = os.path.join('data', 'train_set.txt')
test_file_path_txt = os.path.join('data', 'test_set.txt')
validation_file_path_txt = os.path.join('data', 'validation_set.txt')

# files that contain dataset values
train_dataset_path_json = os.path.join('data', 'train_dataset.json')
test_dataset_path_json = os.path.join('data', 'test_dataset.json')
validation_dataset_path_json = os.path.join('data', 'validation_dataset.json')

# For replication purposes
np.random.seed(1)


# split data into train, test, validation datasets
def split_data():
    data = []
    for wav_folder in os.scandir(f'{cut_path}'):
        data.append(os.path.join(cut_path, os.path.basename(wav_folder.name)))
    np.random.shuffle(data)  # shuffle data
    # determine train and test ending index in data
    train_index_end = int(len(data) * 0.05) # 0.8
    test_index_end = int(len(data) * 0.05) + int(len(data) * 0.05) # 0.8, 0.1
    # get train and test from data
    train = data[:train_index_end]
    test = data[train_index_end:test_index_end]
    validation = test #data[test_index_end:]
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

    # get .wav folders from dataset
    for i, wav_folder_path in enumerate(dataset):
        wav_folder_path = wav_folder_path.replace('\n', '')
        # get sounds from all .wav folders
        for sound_path in os.scandir(os.path.join(wav_folder_path)):
            sound_paths.append(os.path.join(wav_folder_path, sound_path.name))
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
    cepstrum_features = [get_cepstrum_features(sound_path) for sound_path in sound_paths]

    for sound_path, label, cepstrum_feature in zip(sound_paths, labels, cepstrum_features):
        result.append({
            "sound_path": sound_path,
            "label": label,
            "cepstrum_feature": cepstrum_feature.tolist()
        })
    return result


# get cepstrum features from file at file_path
def get_cepstrum_features(sound_path):
    result = None
    try:
        y, sr = librosa.load(sound_path)
        result = librosa.feature.mfcc(y=y, sr=sr, n_fft=512)
        # fig, ax = plt.subplots()
        # img = librosa.display.specshow(result, x_axis='time', ax=ax)
        # fig.colorbar(img, ax=ax)
        # ax.set(title='MFCC')
        # plt.show()
    except Exception as e:
        print(e)
        print(sound_path)
        x = 3 / 0
    return result


# get train/test/validation dataset as json (format: sound_path, label, cepstrum_features)
# write to json file
def write_datasets_to_json():
    train_dataset_file = open(train_dataset_path_json, 'w')
    train_data = get_dataset(train_file_path_txt)
    train = json.dumps(train_data)
    train_dataset_file.write(train)
    train_dataset_file.close()
    print('train_dataset_file finished!')

    test_dataset_file = open(test_dataset_path_json, 'w')
    test_data = get_dataset(test_file_path_txt)
    test = json.dumps(test_data)
    test_dataset_file.write(test)
    test_dataset_file.close()
    print('test_dataset_file finished!')

    validation_dataset_file = open(validation_dataset_path_json, 'w')
    validation_data = get_dataset(validation_file_path_txt)
    validation = json.dumps(validation_data)
    validation_dataset_file.write(validation)
    validation_dataset_file.close()
    print('validation_dataset_file finished!')


# load dataset and return dataset values
def load_dataset_from_json(dataset_path):
    file = open(dataset_path)
    json_data = json.loads(file.readline())
    file.close()
    return json_data

# write_datasets_to_json()
# load_dataset_from_json('test_dataset.json')

# paths = get_dataset_sound_filenames(validation_file_path)
# label = attach_labels_to_sound_paths(paths)[0][0]
# get_cepstrum_features(label)

# path na dataset_folder
# dohvatit sve njegove splittane glasove ->


# dohvatit train,test,validation .wav imena datoteka
# dohvatit njihove cuts (.wav datoteka, label = sound)
# .wav cuts pretvorit u kepstralne znacajke, resize na average shape
