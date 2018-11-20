import glob
import struct
import wave
from collections import Counter
from operator import itemgetter

import librosa
import numpy as np
from tslearn.metrics import dtw


def compute_mfcc_from_file(file):
    time_characteristic = create_time_characteristics_of_a_file(file)
    mfcc = librosa.feature.mfcc(y=time_characteristic, sr=16000, n_mfcc=13)
    return mfcc


def create_time_characteristics_of_a_file(file):
    wave_file = wave.open(file, 'r')
    # rate = wave_file.getframerate()
    length = wave_file.getnframes()
    time_plot = []
    for i in range(0, length):
        wave_data = wave_file.readframes(1)
        data = struct.unpack("<h", wave_data)
        time_plot.append(int(data[0]))
    return np.array(time_plot, dtype=np.float32)


def compute_spectral_roloff(file):
    chars = create_time_characteristics_of_a_file(file)
    return librosa.feature.spectral_rolloff(chars, sr=16000)[0]


def calculate_dict(mfcc_values, rolloff_values, names, labels):
    final_dict = dict()
    for i in names:
        final_dict[i] = []
    for id1, (mf1, ro1, nm1, lb1) in enumerate(zip(mfcc_values, rolloff_values, names, labels)):
        for id2, (mf2, ro2, nm2, lb2) in enumerate(zip(mfcc_values, rolloff_values, names, labels)):
            if id1 < id2:
                current_dtw = dtw(mf1, mf2)
                # current_dtw = dtw(mf1 + ro1, mf2 + ro2)
                final_dict[nm1].append({"name": nm2, "label": lb2, "distance": current_dtw})
                final_dict[nm2].append({"name": nm1, "label": lb1, "distance": current_dtw})
    for final_key, final_item in final_dict.items():
        final_dict[final_key] = sorted(final_item, key=itemgetter('distance'))
        # print(key, len(final_dict[key]))
    return final_dict


def recognize_speech(vector, k=1):
    nearest_neighbours = Counter(elem["label"] for elem in vector[:k])
    return nearest_neighbours.most_common(1)[0][0]


if __name__ == '__main__':
    mfcc_list = []
    rolloff_list = []
    name_list = []
    label_list = []
    for wav_name in glob.glob("./*/*.WAV"):
        mfcc_list.append(compute_mfcc_from_file(wav_name).T)
        rolloff_list.append(compute_spectral_roloff(wav_name))
        name_list.append(wav_name.split("/")[-1])
        label_list.append(wav_name.split("/")[-2])
    dist_dict = calculate_dict(mfcc_list, rolloff_list, name_list, label_list)
    for n in range(1, 11):
        accuracy = 0
        print("KNN for k =", n)
        for key, item in dist_dict.items():
            real = label_list[name_list.index(key)]
            predicted = recognize_speech(item, n)
            # print(key, "Real:", real, "Predicted:", predicted)
            if real == predicted:
                accuracy += 1
        print("Accuracy:", accuracy / len(name_list))
