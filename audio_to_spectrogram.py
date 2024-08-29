import soundfile
import matplotlib.pyplot as plt
import numpy as np
import librosa
from librosa.display import specshow
import os
import IPython

rootdir = "D:\projects\Bird sound\data_audio"
# path = (r"Chlorischloris140580.wav")

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        path = os.path.join(subdir, file)
        audio_signal, sampling_rate = soundfile.read(path)
        window_length = int(0.025 * sampling_rate)
        hop_length = int(0.01 * sampling_rate)
        spectrogram = np.abs(
            librosa.stft(audio_signal, hop_length=hop_length, win_length=window_length)
        )
        specshow(
            librosa.amplitude_to_db(spectrogram, ref=np.max),
            sr=sampling_rate,
            hop_length=hop_length,
        )
        plt.savefig(path.replace("wav", "png"), bbox_inches="tight", pad_inches=0)
        plt.close()
