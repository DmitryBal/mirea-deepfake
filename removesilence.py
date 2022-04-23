import os
import librosa
import soundfile as sf

path = '~/Biden/dataset/wavs/'
newpath = '~/Biden/dataset/cleanedwavs/'
ext = ('.wav')
for files in os.listdir(path):
    if files.endswith(ext):
        #print(files)  
        audio_data = path + files
        y , sr = librosa.load(audio_data)
        yt, _ = librosa.effects.trim(y, top_db=80, frame_length=256, hop_length=64)
        sf.write(newpath+files, yt, sr)
    else:
        continue
