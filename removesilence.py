import os
import librosa
import soundfile as sf

path = '~/Biden/'
ext = ('.wav')
for files in os.listdir(path):
    if files.endswith(ext):
        print(files)  
        audio_data = path + files
        y , sr = librosa.load(audio_data)
        yt, _ = librosa.effects.trim(y, top_db=10, frame_length=256, hop_length=64)
        sf.write(path+files, yt, sr)
    else:
        continue
