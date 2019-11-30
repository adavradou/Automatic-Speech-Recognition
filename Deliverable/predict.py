from keras.models import load_model, Sequential
import numpy as np


def predict(model_filepath,audio_streams):
    model = load_model(model_filepath)
    classes = np.load('classes_conv.npy')
    for audio_stream in audio_streams:
        prob = model.predict(audio_stream)
        index = np.argmax(prob[0])
        return classes[index]


