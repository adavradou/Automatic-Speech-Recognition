from keras.models import load_model, Sequential
import numpy as np


def predict(model_filepath,audio_streams):
    model = load_model(model_filepath)
    classes = np.load('classes_conv.npy')
    i=0
    for audio in audio_streams:
        i+=1
        prob = model.predict(np.array([audio,]))
        index = np.argmax(prob[0])
        print("Word #"+str(i) +" : " +classes[index])



