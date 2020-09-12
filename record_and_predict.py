import os
from pathlib import Path
import sounddevice as sd
from scipy.io.wavfile import write
import librosa   # for audio processing

from Deliverable import signal_processing
from Deliverable import feature_exrtaction
from Deliverable import train_model
from Deliverable import split_to_words
from Deliverable import predict

# Record Audio
fs = 8000
max_seconds = 30
print("Recording Audio...")
myrecording = sd.rec( int(max_seconds * fs), samplerate=fs, channels=1)
input("Press Enter to stop recording...")
sd.stop()

write('./recordings/recording.wav', fs, myrecording)  # Save as WAV file

test_audio_path="./recordings"
# Split Audio to test
split_to_words.splitter("./recordings/recording.wav",test_audio_path+"/predict_set")

model_path="model_conv.hdf5"

processed_predict_files, sr, meanDuration = signal_processing.process_predict_dataset(["predict_set"], Path(test_audio_path))


#predict_features=feature_exrtaction.rawData(processed_predict_files, 8000)
#predict_features=feature_exrtaction.rawDataStretched(processed_predict_files, sr, meanDuration)
predict_features=feature_exrtaction.fourier_transform(processed_predict_files, sr)
#predict_features,feature_size=feature_exrtaction.fourier_transform_stretched(processed_predict_files, sr, meanDuration,feature_size)
#predict_features=feature_exrtaction.extract_mfccs(processed_predict_files, sr, meanDuration)
#predict_features=feature_exrtaction.extract_fourier_peaks(processed_predict_files,sr,meanDuration)

print("Predictions")
# Predict based on trained model
predict.predict(model_path, predict_features, processed_predict_files)
