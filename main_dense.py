# ###
# Libraries
# ###

import os
# os.environ['KERAS_BACKEND'] = 'tensorflow'
import librosa   # for audio processing
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile  # for audio processing
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils, to_categorical
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D, SimpleRNN
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from keras import backend as K
from keras.models import load_model, Sequential
import random
from sklearn.metrics import accuracy_score


# ###
# Data Preparation
# ###

current_directory = os.getcwd()     # Audio Path
train_audio_path = current_directory + '/speech_commands_dataset'
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

# Plot count of labels and duration of digit recordings
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


print('X data:\n', all_wave)    # [array([-0.0003346 , ..., ], dtype=float32), array....]
print('Y data:\n', all_label)   # ['zero', ..., 'nine']
all_wave = np.array(all_wave)
all_label = np.array(all_label)
print('Size of X:\n', all_wave.shape)     # (77, 8000)
print('Size of Y:\n', all_label.shape)    # (77,)
print('X data (as array):\n', all_wave)   # [[-3.3460365e-04 ...][...]...]
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
y_ = np_utils.to_categorical(y_, num_classes=len(labels))     # From int to one-hot
print('to_categorical:\n', y_)
x_tr, x_val, y_tr, y_val = train_test_split(np.array(x_), np.array(y_), stratify=y_, test_size=0.2, random_state=777, shuffle=True)
print('x_tr: ', x_tr.shape, '\nx_val: ', x_val.shape, '\ny_tr: ', y_tr.shape, '\ny_val: ', y_val.shape)
print('y_val:\n', y_val)
n_cols = x_tr.shape[1]

model = Sequential()

# model.add(Dense(100, activation='sigmoid', input_dim=n_cols))
# model.add(Dense(200, activation='sigmoid'))
# model.add(Dense(200, activation='sigmoid'))

# model.add(Dense(80, activation='tanh', input_dim=n_cols))
# model.add(Dense(200, activation='tanh'))
# model.add(Dense(150, activation='tanh'))
# model.add(Dense(200, activation='sigmoid'))

# model.add(Dense(100, activation='relu', input_dim=n_cols))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(500, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(100, activation='sigmoid'))

# model.add(Dense(500, activation='relu', input_dim=n_cols))
# model.add(Dense(500, activation='relu'))
# model.add(Dense(500, activation='relu'))
# model.add(Dense(10, activation='sigmoid'))

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
filepath = current_directory + '/model_dense.hdf5'
mc = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model.save(current_directory + '/model_dense.hdf5', True, True)

# Fit Model
history = model.fit(x_tr, y_tr, epochs=100, callbacks=[es, mc], batch_size=64, validation_data=(x_val, y_val))

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

model = load_model(filepath)

# evaluate model
_, accuracy = model.evaluate(x_tr, y_tr)
print('Accuracy: %.2f' % (accuracy*100))
print(y_val)
y_pred = model.predict_classes(x_tr)


# Predict
def predict(audio):
    prob = model.predict_classes(audio.reshape(1, 8000))
    index = np.argmax(prob[0])
    return classes[index]


for rounds in range(5):
    index = random.randint(0, len(x_val)-1)
    samples = x_val[index].ravel()
    ipd.Audio(samples, rate=8000)
    print("Audio:", classes[np.argmax(y_val[index])], " and predicted:", predict(samples))

# Scores
score_train = model.evaluate(x_tr, y_tr, verbose=0)
score_val = model.evaluate(x_val, y_val, verbose=0)
print('Train Score: ', score_train, '\nValidation Score: ', score_val)

print('\nEND')
