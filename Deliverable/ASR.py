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



current_directory = os.getcwd()

# Choose audio dataset
train_audio_path = Path('../free-spoken-digit-dataset-medium') #Audio Path
#train_audio_path = Path('../speech_commands_dataset_full') #Audio Path

print("Training Dataset Folder:",train_audio_path)

labels = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

# Training
#rawAudioFiles,all_label, sr, meanDuration = signal_processing.resample_dataset(labels, train_audio_path)
processedAudioFiles, all_labels, sr, meanDuration = signal_processing.process_dataset(labels, train_audio_path)


all_features=feature_exrtaction.rawData(processedAudioFiles, sr)
#all_features=feature_exrtaction.rawDataStretched(processedAudioFiles, sr, meanDuration)
#all_features=feature_exrtaction.fourier_transform(processedAudioFiles, sr)
#all_features=feature_exrtaction.fourier_transform_stretched(processedAudioFiles, sr, meanDuration)
#all_features=feature_exrtaction.extract_mfccs(processedAudioFiles, sr, meanDuration)
#all_features=feature_exrtaction.extract_fourier_peaks(processedAudioFiles,sr,meanDuration)

#model_path = train_model.train_dense(all_features,all_labels, labels, model_name="dense_full")
model_path = train_model.train_convolutional(all_features,all_labels, labels, "Just_testing_conv")

test_audio_path="../free-spoken-digit-dataset-medium"

# Predicting Features extraction
#raw_predict_files,all_label, sr, meanDuration = signal_processing.resample_dataset(["predict_set"], Path(test_audio_path))
processed_predict_files, sr, meanDuration = signal_processing.process_predict_dataset(labels, Path(test_audio_path))


predict_features=feature_exrtaction.rawData(processed_predict_files, sr)
#predict_features=feature_exrtaction.rawDataStretched(processed_predict_files, sr, meanDuration)
#predict_features=feature_exrtaction.fourier_transform(processed_predict_files, sr)
#predict_features=feature_exrtaction.fourier_transform_stretched(processed_predict_files, sr, meanDuration)
#predict_features=feature_exrtaction.extract_mfccs(processed_predict_files, sr, meanDuration)
#predict_features=feature_exrtaction.extract_fourier_peaks(processed_predict_files,sr,meanDuration)

print("Predictions")
# Predict based on trained model
predict.predict(model_path, predict_features, processed_predict_files,"conv")

