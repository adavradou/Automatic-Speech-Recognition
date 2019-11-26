import os
from pathlib import Path


import matplotlib.pyplot as plt
from scipy.fftpack import fft
import sounddevice as sd

import librosa   # for audio processing
import librosa.display
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
from keras.models import Model
from keras import backend as K
import IPython.display as ipd
import random



from scipy import signal


def fourierTransform(fs,y,label,no_of_peaks):
    # Number of samplepoints
    N = len(y)
    # sampling period
    T = 1.0 / fs

    yf = fft(y)

    xf = np.linspace(0.0, 1.0/(2.0*T), N/2) * 0.5 # No need for both semi-axis the rfequency domain in Hertz
    yf = 2.0 * np.abs(yf[:N//2]) # The frequency magnitude

    #plt.plot(xf,yf,label=str(label) + ' - file :' + file.name +"N" + str(N)) # plot the fourier transform
    #plt.show()

    # Find dominant freq peak centers and train matching them with their frequency domains
    peaks=[]

    i=0
    step=10 # Peaks greter than step Hz in distance
    for point in yf:
        peaks.append([xf[i],point])
        i+=1

    # Choose greatest magnitude
    peaks=sorted(peaks, key=lambda x: x[1],reverse=True)
    peaks=peaks[:no_of_peaks]

    return xf,yf,N,peaks



# The voiced speech of a typical adult male will have a fundamental frequency from 85 to 180 Hz, and that of a typical adult female from 165 to 255 Hz.
#def bandpassIIRFilter(data, fs, lowcut=300, highcut=3400, order=4):#telephony - i xroia poy prokyptei apo tis anwteres armonikes einai mallon axristi
#def bandpassIIRFilter(data, fs, lowcut=60, highcut=800, order=4):
def bandpassIIRFilter(data, fs, lowcut=85, highcut=255, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='bandpass')
    #b, a = signal.iirfilter(order, [low, high], rs=60, btype='band', analog = True, ftype = 'cheby2')
    y = signal.lfilter(b, a, data)

    return y


current_directory = os.getcwd()

# Choose audio dataset
train_audio_path = Path('./speech_commands_dataset_small') #Audio Path

print("tap",train_audio_path)

labels = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

i=1
durations=[]
sum=0
processedAudioFiles=[]
all_label = []
sr=8000
print("Signal processing")
for label in labels:
    files = list(train_audio_path.glob(label + '/*.wav'))
    print("processing " + label)
    for file in files:
        originalSignal, sr = librosa.load(str(file), sr=8000, mono=True)
        processedSignal = bandpassIIRFilter(originalSignal, sr)
        processedSignal, index = librosa.effects.trim(originalSignal)
        duration = librosa.get_duration(processedSignal)
        sum+=duration
        processedAudioFiles.append(processedSignal) # Stage processing/feature extraction to minimise complexity
        i+=1

        all_label.append(label) # Create labels list

meanDuration=sum/i

sumdur=0
i=0
fsizeEQ=0
fsizeL=0
fsizeG=0


all_features = []
desiredNoOfFeatures=13
# Features Extraction an time normalisation
print("Extracting Features from "+str(i)+" files, mean duration="+str(meanDuration))
for audio in processedAudioFiles:

    #sd.play(audio, sr)
    #status = sd.wait()

    duration = librosa.get_duration(audio)
    sumdur+=duration

    ratio = duration / meanDuration
    if ratio < 0.1:
        ratio = 0.05

    # Stretrch audio to normalise spoken word duration
    audio = librosa.effects.time_stretch(audio, ratio)

    #sd.play(audio, sr)
    #status = sd.wait()

    frequency, fourierMagnitude, sampleCount, peaks = fourierTransform(sr, audio, label, no_of_peaks=66)

# audio <- Raw Signal
# fourierMagnitude <- Fourier representation of the signal (1o tetartimorio)

    all_features.append(fourierMagnitude)

    i+=1


print("Features have been extracted")

all_features= np.array(all_features)

# Neural Network Classification
# Label Encoding and train-test split
le = LabelEncoder()     # Encode labels from (zero, one, ...) to (0, ..., 1)
y = le.fit_transform(all_label)
classes = list(le.classes_)
np.save(current_directory + '/classes_conv.npy', le.classes_)
y = np_utils.to_categorical(y, num_classes=len(labels))     # From int to one-hot
all_features = np.array(all_features).reshape(-1, len(all_features[0]),1)      # only for convolutional nn (2D to 3D)
x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_features), np.array(y), stratify=y, test_size=0.2, random_state=777, shuffle=True)

K.clear_session()

print(all_features[0].shape,len(all_features[0]))

inputs = Input(shape=(len(all_features[0]),1))

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

filepath = current_directory + '/model_fourier_conv.hdf5'

# Keep only a single checkpoint, the best over test accuracy.
mc = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model.save(current_directory + '/model_fourier_conv.hdf5', True, True)

history = model.fit(x_tr, y_tr, epochs=100, callbacks=[es, mc], batch_size=32, validation_data=(x_val, y_val))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

#model = load_model(filepath)

# Predict
def predict(test):
    prob = model.predict(np.array([test, ]))
    index = np.argmax(prob[0])
    return classes[index]


for rounds in range(10):
    index = random.randint(0, len(x_val) - 1)
    sample_file = x_val[index]
    print("Audio:", classes[np.argmax(y_val[index])], " and predicted:", predict(sample_file))

score_train = model.evaluate(x_tr, y_tr, verbose=0)
score_val = model.evaluate(x_val, y_val, verbose=0)

print('Train Score: ', score_train, '\nValidation Score: ', score_val)

print('\nEND')
