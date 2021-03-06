import os

from Deliverable import signal_processing
from Deliverable import feature_exrtaction
from Deliverable import train_model
from Deliverable import split_to_words
from Deliverable import predict

from pathlib import Path
import sounddevice as sd
from scipy.io.wavfile import write
import sys
import librosa   # for audio processing

current_directory = os.getcwd()

# Choose audio dataset
# train_audio_path = Path('../free-spoken-digit-dataset-medium')  # Audio Path
train_audio_path = Path('../speech_commands_dataset_small')   # Audio Path
# train_audio_path = Path('../speech_commands_dataset_full')   # Audio Path
train_audio_path = Path('../free-spoken-digit-dataset-medium') #Audio Path
#train_audio_path = Path('../speech_commands_dataset_full') #Audio Path

print("Training Dataset Folder:", train_audio_path)
print("Training Dataset Folder:",train_audio_path)

labels = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

# Training
# rawAudioFiles,all_label, sr, meanDuration = signal_processing.resample_dataset(labels, train_audio_path)
#rawAudioFiles,all_label, sr, meanDuration = signal_processing.resample_dataset(labels, train_audio_path)
processedAudioFiles, all_labels, sr, meanDuration = signal_processing.process_dataset(labels, train_audio_path)

print('PreProcessing\n1)Raw Data\n2)Raw Data Stretched\n3)Fourier\n4)Fourier Stretched\n5)MFCC\n6)Fourier Peaks\n')
ans_pp = int(input('Select one of the above for data pre-processing\n'))
print('Model Architecture\n1)Dense\n2)Convolutional\n')
ans_m = int(input('Select one of the above for train\n'))

if ans_pp == 1:
    all_features = feature_exrtaction.rawData(processedAudioFiles, sr)
elif ans_pp == 2:
    all_features = feature_exrtaction.rawDataStretched(processedAudioFiles, sr, meanDuration)
elif ans_pp == 3:
    all_features = feature_exrtaction.fourier_transform(processedAudioFiles, sr)
elif ans_pp == 4:
    all_features = feature_exrtaction.fourier_transform_stretched(processedAudioFiles, sr, meanDuration)
elif ans_pp == 5:
    all_features = feature_exrtaction.extract_mfccs(processedAudioFiles, sr, meanDuration)
elif ans_pp == 6:
    all_features = feature_exrtaction.extract_fourier_peaks(processedAudioFiles, sr, meanDuration)
else:
    sys.exit('Invalid selection')

if ans_m == 1:
    model_path = train_model.train_dense(all_features, all_labels, labels, model_name="Just_testing_dense")
elif ans_m == 2:
    model_path = train_model.train_convolutional(all_features, all_labels, labels, "Just_testing_conv")
else:
    sys.exit('Invalid selection')

test_audio_path="../free-spoken-digit-dataset-medium"

# Predicting Features extraction
# raw_predict_files,all_label, sr, meanDuration = signal_processing.resample_dataset(["predict_set"], Path(test_audio_path))
processed_predict_files, all_labels, sr, meanDuration = signal_processing.process_dataset(labels, Path(test_audio_path))

if ans_pp == 1:
    predict_features =feature_exrtaction.rawData(processed_predict_files, sr)
elif ans_pp == 2:
    predict_features = feature_exrtaction.rawDataStretched(processed_predict_files, sr, meanDuration)
elif ans_pp == 3:
    predict_features = feature_exrtaction.fourier_transform(processed_predict_files, sr)
elif ans_pp == 4:
    predict_features = feature_exrtaction.fourier_transform_stretched(processed_predict_files, sr, meanDuration)
elif ans_pp == 5:
    predict_features = feature_exrtaction.extract_mfccs(processed_predict_files, sr, meanDuration)
elif ans_pp == 6:
    predict_features = feature_exrtaction.extract_fourier_peaks(processed_predict_files, sr, meanDuration)
else:
    sys.exit('Error')
#raw_predict_files,all_label, sr, meanDuration = signal_processing.resample_dataset(["predict_set"], Path(test_audio_path))
processed_predict_files, sr, meanDuration = signal_processing.process_predict_dataset(labels, Path(test_audio_path))


print("Predictions")
# Predict based on trained model
if (ans_m==2):
    predict.predict(model_path, predict_features, processed_predict_files,"conv")
else :
    predict.predict(model_path, predict_features, processed_predict_files, "dense")

