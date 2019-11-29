import os
from pathlib import Path

from Deliverable import signal_processing
from Deliverable import feature_exrtaction

current_directory = os.getcwd()

# Choose audio dataset
train_audio_path = Path('./free-spoken-digit-dataset-medium') #Audio Path

print("Training Dataset Folder:",train_audio_path)

labels = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

#rawAudioFiles,all_label, sr, meanDuration = signal_processing.resample_dataset(labels, train_audio_path)
processedAudioFiles, all_label, sr, meanDuration = signal_processing.process_dataset(labels, train_audio_path)


#all_features=feature_exrtaction.rawData(processedAudioFiles, sr)
#all_features=feature_exrtaction.rawDataStretched(processedAudioFiles, meanDuration)
#all_features=feature_exrtaction.fourier_transform(processedAudioFiles, sr)
#all_features=feature_exrtaction.fourier_transform_stretched(processedAudioFiles, sr, meanDuration)
all_features=feature_exrtaction.extract_mfccs(processedAudioFiles, sr, meanDuration)
#all_features=feature_exrtaction.extract_fourier_peaks(processedAudioFiles,sr,meanDuration)

train_model()

predict()
