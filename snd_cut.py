from pydub import AudioSegment
import os
import shutil

wav_path = 'data/original/wav'
lab_path = 'data/original/lab'
cut_path = 'data/cut'

def snd_cut():
    # Delete folder containing sound cuts
    if os.path.exists(f'{cut_path}'):
        shutil.rmtree(f'{cut_path}')

    os.makedirs(f'{cut_path}')

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
            if not os.path.exists(f'{cut_path}/{wav_file.name}'):
                os.makedirs(f'{cut_path}/{wav_file.name}')
            # saving sound in directory dedicated for his parent wav
            filename = f'{i:05d}_{sound}'
            wav_single_sound.export(f'{cut_path}/{wav_file.name}/{filename}.wav', format="wav")
        file.close()

# cut i sejvale cuttano orginizairano


# dohvatit train,test,validation .wav imena datoteka
# dohvatit njihove cuts (.wav datoteka, label = sound)
# .wav cuts pretvorit u kepstralne znacajke, resize na average shape

# cnn model, input shape je onaj gore
# trenirat


# open model
# validiranje