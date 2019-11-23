import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
import librosa   # for audio processing
import IPython.display as ipd
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.models import load_model

current_directory = os.getcwd()
filepath = current_directory + '/model_conv_kostas.hdf5'
model = load_model(filepath)


def predict(audio):
    prob = model.predict(audio.reshape(1, 8000, 1))
    index = np.argmax(prob[0])
    return classes[index]

######## MERGE MULTIPLE AUDIO FILES IN ONE ############
filepath = os.getcwd() + "/testing_numbers/"

#Choose the numbers with the prefered order to test. 
silence = AudioSegment.from_wav(filepath+"silence.wav")
sound1 = AudioSegment.from_wav(filepath+"2.wav")
sound2 = AudioSegment.from_wav(filepath+"4.wav")
sound3 = AudioSegment.from_wav(filepath+"6.wav")
sound4 = AudioSegment.from_wav(filepath+"8.wav")
sound5 = AudioSegment.from_wav(filepath+"7.wav")

#Combine them in one audio file.
combined_sounds = sound1 + silence + sound2 + silence + sound3 + silence + sound4 + silence + sound5
combined_sounds.export("testing_sequence.wav", format="wav")


######### SPLIT THE PHRASE INTO WORDS ############
#sound_file = AudioSegment.from_wav("testing_sequence.wav")
#audio_chunks = split_on_silence(sound_file, 
#    # must be silent for at least half a second
#    min_silence_len=1000,
#
#    # consider it silent if quieter than -36 dBFS
#    silence_thresh=-36
#)
#
##Export each digit of a sequence in a .wav file.
#for i, chunk in enumerate(audio_chunks):
#    out_file = "chunk{0}.wav".format(i)
#    print("exporting", out_file)
#    chunk.export(out_file, format="wav")
#    
#    samples, sample_rate = librosa.load("chunk{0}.wav".format(i), sr = 16000)
#    samples = librosa.resample(samples, sample_rate, 8000)
#    ipd.Audio(samples,rate=8000)  
#    
#    #predict(samples)
#    print("Text:", predict(samples))

