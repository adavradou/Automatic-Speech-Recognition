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


    #sd.play(audio, sr)
    #status = sd.wait()


    i+=1

print("Features have been extracted")

# Neural Network Classification

all_features = np.array(all_features)
all_label = np.array(all_label)
print('Size of X:\n', all_features.shape)     # (77, 8000)
print('Size of Y:\n', all_label.shape)    # (77,)

x_ = all_features
y_ = all_label

le = LabelEncoder()
y_ = le.fit_transform(all_label)
classes = list(le.classes_)

np.save(current_directory + '/classes_dense.npy', le.classes_)
y_ = np_utils.to_categorical(y_, num_classes=len(labels))     # From int to one-hot

