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
    time_characteristic = create_time_characteristics_of_a_file(file)
    mfcc = librosa.feature.mfcc(y=time_characteristic, sr=16000, n_mfcc=5)
    return mfcc


def create_time_characteristics_of_a_file(file):
    waveFile = wave.open(file, 'r')
    rate = waveFile.getframerate()
    length = waveFile.getnframes()
    time_plot = []
    for i in range(0, length):
        waveData = waveFile.readframes(1)
        data = struct.unpack("<h", waveData)
        time_plot.append(int(data[0]))
    return np.array(time_plot, dtype=np.float32)


def compute_spectral_roloff(file):
    chars = create_time_characteristics_of_a_file(file)
    return librosa.feature.spectral_rolloff(chars, sr=16000)[0]


def foo(train_mfccs_, which_mfcc: int, train_roloffs):
    train_mfccs = [mfccs[0] for mfccs in train_mfccs_]
    mfccs_ = []
    for sample in test_set:
        for mfccs, roloff, label in zip(train_mfccs, train_roloffs, labels):
            distance_between_mfccs = dtw(mfccs, compute_mfcc_from_file(sample)[0])
            distance_between_roloffs = dtw(roloff, compute_spectral_roloff(sample))
            mfccs_.append((distance_between_mfccs + distance_between_roloffs, label))
        lul = sorted(mfccs_, key=lambda x: x[0])
        print(lul)
        print(sample, Counter(elem[1] for elem in lul[:7]))


if __name__ == '__main__':
    train_set = []
    knn = KNeighborsClassifier()
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
    train_roloffs = []
    for file in train_set:
        train_roloffs.append(compute_spectral_roloff(file))
        train_mfccs.append(compute_mfcc_from_file(file).T)
    # train_mfccs: first dimension -- how many examples = 252. second dimension -- 2, how many mfcc, third dimension --
    # 24, how many windows
    # print(train_mfccs[0][0], train_mfccs[1][0])
    # print(dtw(train_mfccs[0][0], train_mfccs[1][0]))
    # for i in range(5):
    foo(train_mfccs, 0, train_roloffs)

