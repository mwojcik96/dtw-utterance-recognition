import glob
import struct
import wave
from collections import Counter

import librosa
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from tslearn.metrics import dtw
import random

def compute_mfcc_from_file(file):
    time_characteristic = create_time_characteristics_of_a_file(file)
    mfcc = librosa.feature.mfcc(y=time_characteristic, sr=16000, n_mfcc=13)
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


def recognize_speech(train_mfccs_, train_roloffs):
    accuracy = 0
    for sample in test_set:
        sample_label = sample.split("/")[-2]
        distances = []
        sample_mfcc = compute_mfcc_from_file(sample).T
        # sample_roloff = compute_spectral_roloff(sample)
        for mfccs, roloff, label in zip(train_mfccs_, train_roloffs, labels):
            distance_between_mfccs = dtw(mfccs, sample_mfcc)
            # distance_between_roloffs = dtw(roloff, sample_roloff)
            distances.append((distance_between_mfccs, label))
            # mfccs_.append((distance_between_mfccs + distance_between_roloffs, label))
        dist_list = sorted(distances, key=lambda x: x[0])
        nearest_neighbours = Counter(elem[1] for elem in dist_list[:7])
        print(sample, nearest_neighbours)
        if sample_label == nearest_neighbours.most_common(1)[0][0]:
            accuracy += 1
        print(f"ACCURACY for {test_set[0]}: ", accuracy/len(test_set))


if __name__ == '__main__':
    prefixes = set()
    for file in glob.glob("./*/*.WAV"):
        prefixes.add(file.split("/")[-1][:5])
    # randomly choose ~30% of probes to test_set
    for index, _ in enumerate(glob.glob("./*/*.WAV")):
        train_set = []
        test_set = []
        labels = []
        for index2, example2 in enumerate(glob.glob("./*/*.WAV")):
            if index2 == index:
                test_set.append(example2)
            else:
                train_set.append(example2)
                labels.append(example2.split("/")[-2])
        train_mfccs = []
        train_roloffs = []
        for file in train_set:
            train_roloffs.append(compute_spectral_roloff(file))
            train_mfccs.append(compute_mfcc_from_file(file).T)
        # train_mfccs: first dimension -- how many examples = 252. second dimension -- 24 how many windows,
        # third dimension n --, how many mfcc
        # print(train_mfccs[0][0], train_mfccs[1][0])
        # print(dtw(train_mfccs[0][0], train_mfccs[1][0]))
        # for i in range(5):
        recognize_speech(train_mfccs, train_roloffs)

