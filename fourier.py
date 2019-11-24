import os
from pathlib import Path
import wave

import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft
#from numpy import fft
import numpy as np

import sounddevice as sd


from scipy import signal


def fourier(fs,y,label):
    print("fs:",fs,y,type(y),len(y))
    # Number of samplepoints
    N = len(y)
    # sample spacing
    T = 1.0 / fs
    yf = fft(y)
    xf = np.linspace(70.0, 400.0, N / 2) # No need for negative semi-axis


    fig, ax = plt.subplots()
    ax.plot(xf, 1.0 / N * np.abs(yf[:N //2]))
    ax.set_title(str(label)+' - file :'+file.name)
    plt.show()


#def bandpassIIRFilter(data,fs,lowcut=85,highcut=255,order=4): #maybe amplify result # The voiced speech of a typical adult male will have a fundamental frequency from 85 to 180 Hz, and that of a typical adult female from 165 to 255 Hz.
def bandpassIIRFilter(data,fs,lowcut=60,highcut=400,order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='bandpass',  analog=True)
    #b, a = signal.iirfilter(order, [low, high], rs=60, btype='band', analog = True, ftype = 'cheby2')
    y = signal.lfilter(b, a, data)
    print(len(y),y)

    return y

# Sample rate and desired cutoff frequencies (in Hz).
lowcut = 0.5
highcut = 0.7


current_directory = os.getcwd()

train_audio_path = Path('./noizy_dataset') #Audio Path
#train_audio_path = Path('./speech_commands_dataset-v0.01-small') #Audio Path

print("tap",train_audio_path)

labels = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

duration_of_recordings = []
for label in labels:
    files = list(train_audio_path.glob(label+'/*.wav'))
    for file in files:
        fs, data = wav.read(file)

        # fig, ax = plt.subplots()
        # ax.plot(data)
        # ax.set_title(str(label) + ' - file :' + file.name)
        # plt.show()


        # If the file is stereo convert to mono by getting avg
        w = wave.open(str(file.resolve()), mode='rb')
        if w.getnchannels() == 2:
            #audiodata = data.asFloat()  # Avoid buffer overflow
            #y = audiodata.sum(axis=1) / 2 # combine channels
            y = data.T[0] # Keep One channel
        else:
            y = data


        filtered=bandpassIIRFilter(y,fs)
        sd.play(y, fs)
        status = sd.wait()

        sd.play(filtered, fs)
        status = sd.wait()

        fourier(fs,filtered,label)

    break


