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

def processDataset(labels,train_audio_path):
    i = 1
    durations = []
    sum = 0
    processedAudioFiles = []
    all_label = []
    sr = 8000
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
    return processedAudioFiles,meanDuration

# Neural here
#x_tr, x_val, y_tr, y_val = train_test_split(np.array(x_), np.array(y_), stratify=y_, test_size=0.2, random_state=777, shuffle=True)
