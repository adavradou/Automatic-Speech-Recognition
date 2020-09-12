from keras.models import load_model, Sequential
import numpy as np
import sounddevice as sd

def predict(model_filepath,audio_streams,processed_predict_files,mode):
    model = load_model(model_filepath)
    if (mode == "conv"):
        classes = np.load('classes_conv.npy')
    else:
        classes = np.load('classes_dense.npy')

    print(classes)
    i=0
    for audio in audio_streams:
        if(mode=="conv"):
            prob = model.predict(audio.reshape(1, len(audio), 1))
        else:
            prob = model.predict(np.array([audio, ]))
        index = np.argmax(prob[0])
        print("Word #"+str(i+1) +" : " +classes[index])
        #sd.play(processed_predict_files[i], 8000)
        #status = sd.wait()
        i+=1


