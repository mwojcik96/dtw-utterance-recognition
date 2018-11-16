import glob
import wave, struct
from collections import Counter

import librosa
import matplotlib.pyplot as plt
import numpy as np
import python_speech_features
from sklearn.neighbors import KNeighborsClassifier
from tslearn.metrics import dtw


def compute_mfcc_from_file(file):
    waveFile = wave.open(file, 'r')
    rate = waveFile.getframerate()
    length = waveFile.getnframes()
    time_plot = []
    for i in range(0, length):
        waveData = waveFile.readframes(1)
        data = struct.unpack("<h", waveData)
        time_plot.append(int(data[0]))
    # plt.plot(range(len(time_plot)), np.array(time_plot, dtype=np.float32))
    # plt.show()
    time_characteristic = np.array(time_plot, dtype=np.float32)
    # mfcc_feat = python_speech_features.mfcc(rate, time_characteristic)
    mfcc = librosa.feature.mfcc(y=time_characteristic, sr=16000, n_mfcc=5)
    return mfcc


def foo(train_mfccs, which_mfcc: int):
    train_mfccs = [mfccs[which_mfcc] for mfccs in train_mfccs]
    min_mfcc = 10000000
    min_label = ""
    print(test_set)
    xd = []
    for sample in test_set:
        for mfccs, label in zip(train_mfccs, labels):
            new_nfcc = dtw(mfccs, compute_mfcc_from_file(sample)[0])
            xd.append((new_nfcc, label))
            if new_nfcc < min_mfcc:
                min_mfcc = new_nfcc
                min_label = label
        lul = sorted(xd, key=lambda x: x[0])
        print(lul)
        print(sample, Counter(elem[1] for elem in lul[:7]))


if __name__ == '__main__':
    train_set = []
    test_set = []
    labels = []
    for file in glob.glob("./*/*.WAV"):
        if 'AF1K1' in file:
            test_set.append(file)
        else:
            train_set.append(file)
            labels.append(file.split("/")[-2])
    # print(train_set[0])
    train_mfccs = []
    for file in train_set:
        train_mfccs.append(compute_mfcc_from_file(file))
    # train_mfccs: first dimension -- how many examples = 252. second dimension -- 2, how many mfcc, third dimension --
    # 24, how many windows
    print(train_mfccs)
    print(len(train_mfccs))
    print(len(train_mfccs[0][0]))
    # print(train_mfccs[0][0], train_mfccs[1][0])
    # print(dtw(train_mfccs[0][0], train_mfccs[1][0]))
    for i in range(5):
        foo(train_mfccs, i)

