from pydub import AudioSegment
import os

wav_path = 'data/original/wav'
lab_path = 'data/original/lab'
dest_path = 'data/cut'

# Delete folder containing sound cuts
if os.path.exists(f'{dest_path}'):
    os.rmdir(f'{dest_path}')

os.makedirs(f'{dest_path}')

# Extract wav and lab files
wav_files = os.scandir(f'{wav_path}')
lab_files = os.scandir(f'{lab_path}')

# extracting sounds from .wav files
for wav_file, lab_file in zip(wav_files, lab_files):
    wav = AudioSegment.from_wav(wav_file)
    file = open(lab_file)
    for i, line in enumerate(file):
        # lab file has start,end,sound. extracting this to cut wav files
        line_content = line.split(" ")
        start_time = int(int(line_content[0]) / 10**4)  # time is in microseconds*0.1, we convert this to miliseconds (required for AudioSegment)
        end_time = int(int(line_content[1]) / 10**4)
        sound = line_content[2].replace(':', '-').replace('\n', '')  # Removing ':' and '/n' because we are saving files under sound names and they are invalid chars

        # cutting wav file
        wav_single_sound = wav[start_time:end_time]
        if not os.path.exists(f'{dest_path}/{wav_file.name}'):
            os.makedirs(f'{dest_path}/{wav_file.name}')
        # saving sound in directory dedicated for his parent wav
        filename = f'{i:05d}_{sound}'
        wav_single_sound.export(f'{dest_path}/{wav_file.name}/{filename}.wav', format="wav")
    file.close()

