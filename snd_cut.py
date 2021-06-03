from pydub import AudioSegment
import os
import shutil
import numpy as np

wav_path = os.path.join('data', 'original', 'wav')
lab_path = os.path.join('data', 'original', 'lab')
cut_path = os.path.join('data', 'cut')
sounds_to_ignore = ['buka', 'greska', 'uzdah']


def snd_cut():
    # Delete folder containing sound cuts
    if os.path.exists(f'{cut_path}'):
        shutil.rmtree(f'{cut_path}')

    os.makedirs(f'{cut_path}')
    # Extract wav and lab files
    wav_files = os.scandir(f'{wav_path}')
    desired_duration = 0

    # calculate desired_duration based on average/max? sound lengths
    for lab_file_name in os.scandir(lab_path):
        file = open(lab_file_name)
        durations = []  # durations of audio files (ms) in lab files
        for line in file:
            line_content = line.split(" ")
            start_time = int(int(line_content[
                                     0]) / 10 ** 4)  # time is in microseconds*0.1, we convert this to miliseconds (required for AudioSegment)
            end_time = int(int(line_content[1]) / 10 ** 4)
            durations.append(end_time - start_time)
        file.close()

    print(f'DURATIONS min:{np.min(durations)} max:{np.max(durations)} average:{np.average(durations)} median:{np.median(durations)}')
    # DURATIONS min:24 max:376 average:69.76744186046511 median:56.0
    desired_duration = np.round(np.average(durations))

    # extracting sounds from .wav files
    for wav_file in wav_files:
        lab_file_name = os.path.join(lab_path, f'{wav_file.name[:-4]}.lab')
        if not os.path.exists(lab_file_name):
            print(f'Lab file at {lab_file_name} not found!')
            continue

        wav = AudioSegment.from_wav(wav_file)
        lab_file = open(lab_file_name)
        for i, line in enumerate(lab_file):
            # lab file has start,end,sound. extracting this to cut wav files
            line_content = line.split(" ")
            start_time = int(int(line_content[
                                     0]) / 10 ** 4)  # time is in microseconds*0.1, we convert this to miliseconds (required for AudioSegment)
            end_time = int(int(line_content[1]) / 10 ** 4)
            duration = end_time - start_time  # duration of audio file in ms

            # if length is 0, then continue (we work with int, not float)
            if start_time == end_time:
                continue

            sound = line_content[2].replace(':', '-').replace('\n', '')  # Removing ':' and '/n' because we are saving files under sound names and they are invalid chars

            if sound in sounds_to_ignore:
                # print(f'Ignoring sound {sound} from {lab_file_name}!')
                continue

            # cutting wav file
            if duration > desired_duration:
                end_time = start_time + desired_duration
                duration = end_time - start_time
            wav_single_sound = wav[start_time:end_time]

            silence = AudioSegment.silent(duration=desired_duration - duration)
            wav_single_sound = wav_single_sound + silence  # Adding silence after the audio

            # windows folder naming is case insensitive
            # we add '_' prefix to uppercase sound labels so they dont merge with the lowercase sound label folder
            sound_folder_name = sound if sound.islower() else f'-{sound}'

            if not os.path.exists(os.path.join(cut_path, sound_folder_name)):
                os.makedirs(os.path.join(cut_path, sound_folder_name))
            # saving sound wav in directory dedicated for his sound label
            filename = f'{wav_file.name[:-4]}-{i:05d}_{sound}'

            wav_single_sound.export(os.path.join(cut_path, sound_folder_name, f'{filename}.wav'), format="wav")
        lab_file.close()


# snd_cut()
