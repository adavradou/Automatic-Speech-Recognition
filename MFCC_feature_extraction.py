import os
from pathlib import Path
import wave

import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft
import scipy.fftpack
#from numpy import fft
import numpy as np
import librosa
import sounddevice as sd

import librosa   # for audio processing
import IPython.display as ipd
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



from scipy import signal


def fourierTransform(fs,y,label):
    # Number of samplepoints
    N = len(y)
    # sampling period
    T = 1.0 / fs

    yf = fft(y)

    xf = np.linspace(0.0, 1.0/(2.0*T), N/2) * 0.5 # No need for both semi-axis the rfequency domain in Hertz
    yf = 2.0/N * np.abs(yf[:N//2]) # The frequency magnitude

    plt.plot(xf,yf,label=str(label) + ' - file :' + file.name +"N" + str(N)) # plot the fourier transform
    plt.show()

    return xf,yf,N



# The voiced speech of a typical adult male will have a fundamental frequency from 85 to 180 Hz, and that of a typical adult female from 165 to 255 Hz.
#def bandpassIIRFilter(data, fs, lowcut=300, highcut=3400, order=4):#telephony - i xroia poy prokyptei apo tis anwteres armonikes einai mallon axristi
def bandpassIIRFilter(data, fs, lowcut=60, highcut=800, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='bandpass')
    #b, a = signal.iirfilter(order, [low, high], rs=60, btype='band', analog = True, ftype = 'cheby2')
    y = signal.lfilter(b, a, data)

    return y

# Sample rate and desired cutoff frequencies (in Hz).
lowcut = 0.5
highcut = 0.7


current_directory = os.getcwd()

train_audio_path = Path('./noizy_dataset') #Audio Path
train_audio_path = Path('./speech_commands_dataset_small') #Audio Path
train_audio_path = Path('./speech_commands_dataset_medium') #Audio Path
#train_audio_path = Path('./speech_commands_dataset-v0.01-small') #Audio Path

print("tap",train_audio_path)

labels = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

all_features = []
all_label = []
i=0
for label in labels:
    files = list(train_audio_path.glob(label+'/*.wav'))
    for file in files:
        # Load and Resample all files to 8k and convert to mono
        #print(str(file.parent)+'/'+str(file.name))
        data, sr = librosa.load(str(file), sr=8000, mono=True)

        # sd.play(data, fs)
        # status = sd.wait()

        filtered = bandpassIIRFilter(data, sr)

        # sd.play(filtered, fs)
        # status = sd.wait()

        mfccs = librosa.feature.mfcc(y=filtered, sr=sr) # Generate mfccs from a time series
        #mfccs = librosa.feature.mfcc(y=filtered, sr=sr, hop_length=1024, htk=True)  # Using a different hop length and HTK-style Mel frequencies

        S = librosa.feature.melspectrogram(y=filtered, sr=sr, n_mels=13,fmax=800) # Use a pre-computed log-power Mel spectrogram
        mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S))

        #mfccs = librosa.feature.mfcc(y=filtered, sr=sr, n_mfcc=13) # Get specific number of components

        # Visualize the MFCC series
        #librosa.display.specshow(mfccs, x_axis='time')
        #plt.colorbar()
        #plt.title('MFCC')
        #plt.tight_layout()
        #plt.show()

        # Compare different DCT bases
        #m_slaney = librosa.feature.mfcc(y=filtered, sr=sr, dct_type=2)
        ##m_htk = librosa.feature.mfcc(y=y, sr=sr, dct_type=3)

        # Visualization
        #plt.figure(figsize=(10, 6))
        #plt.subplot(2, 1, 1)
        #librosa.display.specshow(m_slaney, x_axis='time')
        #plt.title('RASTAMAT / Auditory toolbox (dct_type=2)')
        #plt.colorbar()
        #plt.subplot(2, 1, 2)
        #librosa.display.specshow(m_htk, x_axis='time')
        #plt.title('HTK-style (dct_type=3)')
        #plt.colorbar()
        #plt.tight_layout()
        #plt.show()
        mfccs_flat=mfccs.flatten()
        mfccs_flat=mfccs_flat[:100]
        mean_mfccs=mfccs.mean(axis=1)
        ##########all_features.append(features)
        #print("MFCCS",mfccs.shape)
        ##########all_label.append(label)

        # LPC
        lpc=librosa.lpc(filtered, 16)
        lpc_and_mfcc = np.concatenate((mean_mfccs, lpc), axis=0)
        lpc_and_mfcc_flat = np.concatenate((mfccs_flat, lpc), axis=0)


        #print("LPC+MFCC", mfccs_flat.shape)

        all_features.append(lpc_and_mfcc_flat)
        all_label.append(label)
        if(i==0):
            print(all_features[0].shape)
        i+=1

print("Features are extracted")



#print('X data:\n', all_features)    # [array([-0.0003346 , ..., ], dtype=float32), array....]
#print('Y data:\n', all_label)   # ['zero', ..., 'nine']
all_features = np.array(all_features)
all_label = np.array(all_label)
print('Size of X:\n', all_features.shape)     # (77, 8000)
print('Size of Y:\n', all_label.shape)    # (77,)
#print('X data (as array):\n', all_features)   # [[-3.3460365e-04 ...][...]...]
#print('Y data (as array):\n', all_label)  # ['zero' ... 'nine']


# sc = StandardScaler()
# x_ = sc.fit_transform(all_features)
x_ = all_features
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
#print('y_val:\n', y_val)
n_cols = x_tr.shape[1]

model = Sequential()

model.add(Dense(int(n_cols*2), activation='sigmoid', input_dim=n_cols))
#model.add(Dropout(0.5))
#model.add(Dense(60, activation='relu'))
#model.add(Dropout(0.5))
# model.add(Dense(500, activation='relu'))
model.add(Dense(10, activation='sigmoid'))

model.add(Dense(len(labels), activation='softmax', name='pred'))

# Model Summary
model.summary()

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# Early Stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40, min_delta=0.05)
# Keep only a single checkpoint, the best over test accuracy.
filepath = current_directory + '/model_MFCC.hdf5'
mc = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model.save(current_directory + '/model_MFCC.hdf5', True, True)

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
#print(y_val)
y_pred = model.predict_classes(x_tr)


# Scores
score_train = model.evaluate(x_tr, y_tr, verbose=0)
score_val = model.evaluate(x_val, y_val, verbose=0)

print('Train Score: ', score_train, '\nValidation Score: ', score_val)

print('\nEND')


# Predict
def predict(test):
    prob = model.predict(np.array([test, ]))
    index = np.argmax(prob[0])
    return classes[index]


for rounds in range(10):
    index = random.randint(0, len(x_val) - 1)
    sample_file = x_val[index]
    print("Audio:", classes[np.argmax(y_val[index])], " and predicted:", predict(sample_file))



plt.plot(title='placeholder')
plt.show()
