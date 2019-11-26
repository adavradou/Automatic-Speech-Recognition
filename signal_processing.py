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
train_audio_path = Path('./free-spoken-digit-dataset-medium') #Audio Path

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
        processedAudioFiles.append(processedSignal)
        i+=1
        all_label.append(label)
        if sr != 8000:
            print(sr)

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

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=desiredNoOfFeatures,
                                 hop_length=178)  # Get specific number of components
    mfccs_flat = mfccs.flatten()

    #sd.play(audio, sr)
    #status = sd.wait()

    featuresSize = desiredNoOfFeatures * mfccs.shape[1]

    # print("LPC+MFCC", mfccs_flat.shape)

    if len(mfccs_flat) < featuresSize:
        zero_padded = np.lib.pad(mfccs_flat, ((0), (featuresSize - len(mfccs_flat))), 'constant', constant_values=(1))
        all_features.append(zero_padded)
        fsizeL += 1
        print(len(mfccs_flat) - featuresSize, mfccs.shape, len(audio), duration)
        # all_features.append(mfccs_flat)
    elif len(mfccs_flat) > featuresSize:
        all_features.append(mfccs_flat[:featuresSize])
        fsizeG += 1
        print(len(mfccs_flat) - featuresSize, mfccs.shape, len(audio), duration)
    else:
        fsizeEQ += 1
        all_features.append(mfccs_flat)
        # print(len(mfccs_flat) - featuresSize, mfccs.shape,len(audio),duration)
        # all_features.append(m_htk_flat)
    i+=1


# Neural Network Classification
meanDuration=sumdur/i
print("Extracting Features from "+str(i)+" files, trimmed duration="+str(meanDuration)+" < "+ str(fsizeL)+" = " + str(fsizeEQ) + " > " + str(fsizeG))
print("all_features[0]",all_features[0],all_features[0].shape)
print("Features have been extracted")



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

model.add(Dense(int(n_cols*1.5), activation='sigmoid', input_dim=n_cols))
#model.add(Dropout(0.5))
#model.add(Dense(int(n_cols*1.2), activation='sigmoid'))
model.add(Dense(int(n_cols*0.75), activation='sigmoid'))
#model.add(Dropout(0.5))
# model.add(Dense(500, activation='relu'))
model.add(Dense(30, activation='sigmoid'))

model.add(Dense(len(labels), activation='softmax', name='pred'))

# Model Summary
model.summary()

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# Early Stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40, min_delta=0.05)
# Keep only a single checkpoint, the best over test accuracy.
filepath = current_directory + '/MFCC_stretched('+str(train_audio_path)+'.hdf5'
mc = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model.save(current_directory + '/MFCC_stretched('+str(train_audio_path)+'.hdf5', True, True)

# Fit Model
history = model.fit(x_tr, y_tr, epochs=100, callbacks=[es, mc], batch_size=64, validation_data=(x_val, y_val))

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

#model = load_model(filepath)

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
