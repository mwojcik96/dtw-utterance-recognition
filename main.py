import wave, struct

import librosa
import matplotlib.pyplot as plt
import numpy as np
import python_speech_features

waveFile = wave.open("AK1C1NIE.WAV", 'r')
rate = waveFile.getframerate()
length = waveFile.getnframes()
time_plot = []
for i in range(0,length):
    waveData = waveFile.readframes(1)
    data = struct.unpack("<h", waveData)
    time_plot.append(int(data[0]))

# plt.plot(range(len(time_plot)), np.array(time_plot, dtype=np.float32))
# plt.show()

time_characteristic = np.array(time_plot, dtype=np.float32)
print(time_characteristic.shape)
# mfcc_feat = python_speech_features.mfcc(rate, time_characteristic)
mfcc = librosa.feature.mfcc(y=time_characteristic, sr=16000, n_mfcc=40)
print(mfcc[0])