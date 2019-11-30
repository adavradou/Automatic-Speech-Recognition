import os
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import librosa   # for audio processing
import IPython.display as ipd
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.models import load_model


########## SPLIT THE PHRASE INTO WORDS ############

def splitter(filename,directory):
    sound_file = AudioSegment.from_wav(filename)
    audio_chunks = split_on_silence(sound_file,
                                    # must be silent for at least half a second
                                    min_silence_len=200,

                                    # consider it silent if quieter than -64 dBFS
                                    silence_thresh=-64
                                    )

    # Export each digit of a sequence in a .wav file.
    for i, chunk in enumerate(audio_chunks):
        out_file = directory+"/chunk{0}.wav".format(i)
        print("exporting", out_file)
        chunk.export(out_file, format="wav")


