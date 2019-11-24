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
    b, a = signal.butter(order, [low, high], sr=8000, btype='bandpass')
    #b, a = signal.iirfilter(order, [low, high], rs=60, btype='band', analog = True, ftype = 'cheby2')
    y = signal.lfilter(b, a, data)

    return y

# Sample rate and desired cutoff frequencies (in Hz).
lowcut = 0.5
highcut = 0.7


current_directory = os.getcwd()

train_audio_path = Path('./noizy_dataset') #Audio Path
train_audio_path = Path('./speech_commands_dataset_small') #Audio Path
#train_audio_path = Path('./speech_commands_dataset-v0.01-small') #Audio Path

print("tap",train_audio_path)

labels = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

duration_of_recordings = []
for label in labels:
    files = list(train_audio_path.glob(label+'/*.wav'))
    for file in files:
        # Load and Resample all files to 8k and convert to mono
        print(str(file.parent)+'/'+str(file.name))
        data, fs = librosa.load(str(file), mono=True)
        #y=librosa.to_mono(data)

        #frequency,fourierMagnitude,sampleCount = fourierTransform(fs,data,label)

        #sd.play(data, fs)
        #status = sd.wait()

        filtered=bandpassIIRFilter(data,fs)

        #sd.play(filtered, fs)
        #status = sd.wait()

        frequency,fourierMagnitude,sampleCount = fourierTransform(fs,filtered,label)

    break

plt.plot(title='placeholder')
plt.show()
