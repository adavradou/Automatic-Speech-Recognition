import os
from pathlib import Path
import sounddevice as sd
from scipy.io.wavfile import write


from Deliverable import signal_processing
from Deliverable import feature_exrtaction
from Deliverable import train_model
from Deliverable import split_to_words
from Deliverable import predict



current_directory = os.getcwd()

# Choose audio dataset
train_audio_path = Path('../free-spoken-digit-dataset-medium') #Audio Path
#train_audio_path = Path('../speech_commands_dataset_small') #Audio Path

print("Training Dataset Folder:",train_audio_path)

labels = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

# Training
#rawAudioFiles,all_label, sr, meanDuration = signal_processing.resample_dataset(labels, train_audio_path)
processedAudioFiles, all_labels, sr, meanDuration = signal_processing.process_dataset(labels, train_audio_path)


#all_features=feature_exrtaction.rawData(processedAudioFiles, sr)
#all_features=feature_exrtaction.rawDataStretched(processedAudioFiles, sr, meanDuration)
#all_features=feature_exrtaction.fourier_transform(processedAudioFiles, sr)
all_features=feature_exrtaction.fourier_transform_stretched(processedAudioFiles, sr, meanDuration)
#all_features=feature_exrtaction.extract_mfccs(processedAudioFiles, sr, meanDuration)
#all_features=feature_exrtaction.extract_fourier_peaks(processedAudioFiles,sr,meanDuration)

model_path = train_model.train_dense(all_features,all_labels, labels, model_name="Just_testing_dense")
#model_path = train_model.train_convolutional(all_features,all_labels, labels, "Just_testing_conv")

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
#split_to_words.splitter("recording.wav",test_audio_path+"/predict_set")

# Predicting Features extraction
#raw_predict_files,all_label, sr, meanDuration = signal_processing.resample_dataset(["predict_set"], Path(test_audio_path))
processed_predict_files, all_labels, sr, meanDuration = signal_processing.process_dataset(["predict_set"], Path(test_audio_path))


#predict_features=feature_exrtaction.rawData(processed_predict_files, sr)
#predict_features=feature_exrtaction.rawDataStretched(processed_predict_files, sr, meanDuration)
#predict_features=feature_exrtaction.fourier_transform(processed_predict_files, sr)
predict_features=feature_exrtaction.fourier_transform_stretched(processed_predict_files, sr, meanDuration)
#predict_features=feature_exrtaction.extract_mfccs(processed_predict_files, sr, meanDuration)
#predict_features=feature_exrtaction.extract_fourier_peaks(processed_predict_files,sr,meanDuration)

print("Predictions")
# Predict based on trained model
predict.predict(model_path, predict_features)

