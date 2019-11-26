import os
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.fftpack import fft
import sounddevice as sd

import librosa  # for audio processing
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils, to_categorical
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D, SimpleRNN
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from keras.models import load_model, Sequential
import random
import sounddevice as sd

from scipy import signal


def sortSecond(val):
    return val[1]


def fourierTransform(fs, y, label, no_of_peaks):
    # Number of samplepoints
    N = len(y)
    # sampling period
    T = 1.0 / fs

    yf = fft(y)

    xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2) * 0.5  # No need for both semi-axis the rfequency domain in Hertz
    yf = 2.0 * np.abs(yf[:N // 2])  # The frequency magnitude

    # plt.plot(xf,yf,label=str(label) + ' - file :' + file.name +"N" + str(N)) # plot the fourier transform
    # plt.show()

    # Find dominant freq peak centers and train matching them with their frequency domains
    peaks = []

    i = 0
    step = 10  # Peaks greter than step Hz in distance
    for point in yf:
        peaks.append([xf[i], point])
        i += 1

    # Choose greatest magnitude
    peaks = sorted(peaks, key=lambda x: x[1], reverse=True)
    peaks = peaks[:no_of_peaks]

    return xf, yf, N, peaks


# The voiced speech of a typical adult male will have a fundamental frequency from 85 to 180 Hz, and that of a typical adult female from 165 to 255 Hz.
# def bandpassIIRFilter(data, fs, lowcut=300, highcut=3400, order=4):#telephony - i xroia poy prokyptei apo tis anwteres armonikes einai mallon axristi
def bandpassIIRFilter(data, fs, lowcut=60, highcut=800, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='bandpass')
    # b, a = signal.iirfilter(order, [low, high], rs=60, btype='band', analog = True, ftype = 'cheby2')
    y = signal.lfilter(b, a, data)

    return y


# Sample rate and desired cutoff frequencies (in Hz).
lowcut = 0.5
highcut = 0.7

current_directory = os.getcwd()

train_audio_path = Path('./noizy_dataset')  # Audio Path
# train_audio_path = Path('./testing_numbers') #Audio Path
# train_audio_path = Path('./speech_commands_dataset_medium') #Audio Path
# train_audio_path = Path('./medium_dataset') #Audio Path
train_audio_path = Path('./speech_commands_dataset_small')  # Audio Path
# train_audio_path = Path('./free-spoken-digit-dataset-medium')
# train_audio_path = Path('./speech_commands_dataset-v0.01-small') #Audio Path

print("tap", train_audio_path)

labels = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

all_wave = []
all_label = []
i = 0

for label in labels:
    files = list(train_audio_path.glob(label + '/*.wav'))
    for file in files:
        # Load and Resample all files to 8k and convert to mono
        # print(str(file.parent)+'/'+str(file.name))
        originalSignal, fs = librosa.load(str(file), sr=8000, mono=True)
        data, index = librosa.effects.trim(originalSignal)
        duration = librosa.get_duration(data)
        if duration > 0.35:
            ratio = 0.35 / duration
        else:
            ratio = duration / 0.35

        data = librosa.effects.time_stretch(data, ratio)

        featuresSize = fs
        # y=librosa.to_mono(data)
        # print(len(data),fs)

        # Check filtering
        # frequency,fourierMagnitude,sampleCount = fourierTransform(fs,data,label)

        # sd.play(data, fs)
        # status = sd.wait()

        filtered = bandpassIIRFilter(data, fs)

        # sd.play(filtered, fs)
        # status = sd.wait()

        frequency, fourierMagnitude, sampleCount, peaks = fourierTransform(fs, filtered, label, no_of_peaks=66)

        # create feature set

        # Fourier raw magnitude
        ##print(fourierMagnitude.shape)
        ## Cutoff frequencies > 2000
        # all_wave.append(fourierMagnitude[:2000])
        # all_label.append(label)

        # Peaks only
        # flattened_peaks = [item for sublist in peaks for item in sublist]
        # all_wave.append(flattened_peaks)
        # all_label.append(label)

        # Peaks and fourier raw magnitude
        # flattened_peaks = [item for sublist in peaks for item in sublist]
        # all_wave.append(np.concatenate((flattened_peaks,fourierMagnitude[:1500]),axis=0))
        # all_label.append(label)

        # Peaks Distance
        peak_frequency_domains = []
        i = 0
        prev_peak = 0
        for peak in peaks:
            if i >= 1:
                peak_frequency_domains.append(peak[0] - prev_peak[0])
            else:
                pass
            prev_peak = peak
            i += 1

        flattened_peaks = [item for sublist in peaks for item in sublist]
        all_wave.append(peak_frequency_domains)
        # all_wave.append(np.concatenate((flattened_peaks, peak_frequency_domains), axis=0))

        # if len(data) < featuresSize:
        #    all_wave.append(np.lib.pad(data, ((0),(featuresSize - len(data))), 'constant', constant_values=(0)))
        # else:
        #    all_wave.append(data[:featuresSize])

        all_label.append(label)

# print('X data:\n', all_wave)    # [array([-0.0003346 , ..., ], dtype=float32), array....]
# print('Y data:\n', all_label)   # ['zero', ..., 'nine']
all_wave = np.array(all_wave)
all_label = np.array(all_label)
print('Size of X:\n', all_wave.shape)  # (77, 8000)
print('Size of Y:\n', all_label.shape)  # (77,)
print('X data (as array):\n', all_wave)  # [[-3.3460365e-04 ...][...]...]
print('Y data (as array):\n', all_label)  # ['zero' ... 'nine']

# sc = StandardScaler()
# x_ = sc.fit_transform(all_wave)
x_ = all_wave
y_ = all_label
# print('Standard Scalar:\n', x_)
le = LabelEncoder()
y_ = le.fit_transform(all_label)
classes = list(le.classes_)
# print('Label Encoder:\n', y_)
# print('classes list:\n', classes)
np.save(current_directory + '/classes_dense.npy', le.classes_)
y_ = np_utils.to_categorical(y_, num_classes=len(labels))  # From int to one-hot
print('to_categorical:\n', y_)
x_tr, x_val, y_tr, y_val = train_test_split(np.array(x_), np.array(y_), stratify=y_, test_size=0.2, random_state=777,
                                            shuffle=True)
print('x_tr: ', x_tr.shape, '\nx_val: ', x_val.shape, '\ny_tr: ', y_tr.shape, '\ny_val: ', y_val.shape)
print('y_val:\n', y_val)
n_cols = x_tr.shape[1]

model = Sequential()

model.add(Dense(512, activation='relu', input_dim=n_cols))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
# model.add(Dense(500, activation='relu'))
# model.add(Dense(10, activation='sigmoid'))

model.add(Dense(len(labels), activation='softmax', name='pred'))

# Model Summary
model.summary()

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# Early Stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.1)
# Keep only a single checkpoint, the best over test accuracy.
filepath = current_directory + '/fourier_raw(' + str(train_audio_path) + '.hdf5'
mc = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model.save(current_directory + '/fourier_raw(' + str(train_audio_path) + '.hdf5', True, True)

# Fit Model
history = model.fit(x_tr, y_tr, epochs=100, callbacks=[es, mc], batch_size=64, validation_data=(x_val, y_val))

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# plt.plot(history.history['acc'], label='train')
# plt.plot(history.history['val_acc'], label='test')
# plt.title('Accuracy')
# plt.legend()
# plt.show()

# model = load_model(filepath)

# evaluate model
_, accuracy = model.evaluate(x_tr, y_tr)
print('Accuracy: %.2f' % (accuracy * 100))
print(y_val)
y_pred = model.predict_classes(x_tr)


# Predict
# Predict
def predict(test):
    prob = model.predict(np.array([test, ]))
    index = np.argmax(prob[0])
    return classes[index]


for rounds in range(10):
    index = random.randint(0, len(x_val) - 1)
    sample_file = x_val[index]
    print("Audio:", classes[np.argmax(y_val[index])], " and predicted:", predict(sample_file))

# Scores
score_train = model.evaluate(x_tr, y_tr, verbose=0)
score_val = model.evaluate(x_val, y_val, verbose=0)
print('Train Score: ', score_train, '\nValidation Score: ', score_val)

print('\nEND')