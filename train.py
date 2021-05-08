import data_processing as dp
import os

if not os.path.exists(f'{dp.train_file_path_txt}') or not os.path.exists(f'{dp.test_file_path_txt}'):
    dp.split_and_write_dataset_paths()

if not os.path.exists(f'{dp.train_dataset_path_json}') or not os.path.exists(f'{dp.test_dataset_path_json}'):
    dp.write_datasets_to_json()


def main():
    train = dp.load_dataset_from_json(dp.train_dataset_path_json)
    test = dp.load_dataset_from_json(dp.test_dataset_path_json)

    pass


main()

