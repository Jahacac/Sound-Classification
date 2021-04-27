import data_processing as dp
import os

if not os.path.exists(f'{dp.train_file_path}') or not os.path.exists(f'{dp.test_file_path}'):
    dp.split_and_write_dataset()

def main():
    #print(dp.get_dataset(dp.train_file_path)) this throws: Input signal length=0 is too small to resample from 16000->22050
    pass

main()

