"""
Created on Sep 26, 2019

@author: agapi
"""

import os
# os.environ['KERAS_BACKEND'] = 'tensorflow'
import librosa   # for audio processing
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile  # for audio processing
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.models import load_model
import random


# Audio Path and create labels
current_directory = os.getcwd()
train_audio_path = current_directory + './speech_commands_dataset_small'
# train_audio_path = current_directory + '/speech_commands_dataset_TEST'
dataset_labels = os.listdir(train_audio_path)

# Find count of each label
no_of_recordings = []
for label in dataset_labels:
    waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
    no_of_recordings.append(len(waves))

# Find duration of recording of each digit label
labels = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
duration_of_recordings = []
for label in labels:
    waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
    for wav in waves:
        sample_rate, samples = wavfile.read(train_audio_path + '/' + label + '/' + wav)
        duration_of_recordings.append(float(len(samples)/sample_rate))

# Plot
plt.figure(figsize=(25, 10))
index = np.arange(len(dataset_labels))
plt.subplot(121)
plt.bar(dataset_labels, no_of_recordings)
plt.xticks(index, dataset_labels, fontsize=15, rotation=60)
plt.subplot(122)
plt.hist(np.array(duration_of_recordings))
plt.suptitle('Count of each label and duration of digit labels', y=1)
plt.show()


# Resampling every .wav file from digit labels
all_wave = []
all_label = []
for label in labels:
    print(label)
    waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr=16000)
        samples = librosa.resample(samples, sample_rate, 8000)
        if len(samples) == 8000:
            all_wave.append(samples)
            all_label.append(label)

# Label Encoding and train-test split
le = LabelEncoder()     # Encode labels from (zero, one, ...) to (0, ..., 1)
y = le.fit_transform(all_label)
classes = list(le.classes_)
np.save(current_directory + '/classes_conv.npy', le.classes_)
y = np_utils.to_categorical(y, num_classes=len(labels))     # From int to one-hot
all_wave = np.array(all_wave).reshape(-1, 8000, 1)      # only for convolutional nn (2D to 3D)
x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave), np.array(y), stratify=y, test_size=0.2, random_state=777, shuffle=True)

K.clear_session()

inputs = Input(shape=(8000, 1))

# First Conv1D layer
conv = Conv1D(8, 13, padding='valid', activation='relu', strides=1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

# Second Conv1D layer
conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

# Third Conv1D layer
conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

# Fourth Conv1D layer
conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

# Flatten layer
conv = Flatten()(conv)

# Dense Layer 1
conv = Dense(256, activation='relu')(conv)
conv = Dropout(0.3)(conv)

# Dense Layer 2
conv = Dense(128, activation='relu')(conv)
conv = Dropout(0.3)(conv)

outputs = Dense(len(labels), activation='softmax')(conv)

model = Model(inputs, outputs)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)

filepath = current_directory + '/model_conv.hdf5'

# Keep only a single checkpoint, the best over test accuracy.
mc = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model.save(current_directory + '/model_conv.hdf5', True, True)

history = model.fit(x_tr, y_tr, epochs=100, callbacks=[es, mc], batch_size=32, validation_data=(x_val, y_val))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

model = load_model(filepath)


def predict(audio):
    prob = model.predict(audio.reshape(1, 8000, 1))
    index = np.argmax(prob[0])
    return classes[index]


index = random.randint(0, len(x_val)-1)
samples = x_val[index].ravel()
print("Audio:", classes[np.argmax(y_val[index])])
ipd.Audio(samples, rate=8000)
print("Text:", predict(samples))

score_train = model.evaluate(x_tr, y_tr, verbose=0)
score_val = model.evaluate(x_val, y_val, verbose=0)
print('Train Score: ', score_train, '\nValidation Score: ', score_val)

print('\nEND')
