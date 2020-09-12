import os
from pathlib import Path
import librosa   # for audio processing


from Deliverable import signal_processing
from Deliverable import feature_exrtaction
from Deliverable import predict


labels = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
test_audio_path="../free-spoken-digit-dataset-medium"
test_audio_path="../speech_commands_dataset_full"
test_audio_path="./recordings"

model_path="./conv_Just_testing_conv.hdf5"

# Predicting Features extraction
#raw_predict_files,all_label, sr, meanDuration = signal_processing.resample_dataset(["predict_set"], Path(test_audio_path))
processed_predict_files, sr, meanDuration = signal_processing.process_predict_dataset(["george_set"], Path(test_audio_path))


predict_features=feature_exrtaction.rawData(processed_predict_files, sr)
#predict_features=feature_exrtaction.rawDataStretched(processed_predict_files, sr, meanDuration)
#predict_features=feature_exrtaction.fourier_transform(processed_predict_files, sr)
#predict_features,feature_size=feature_exrtaction.fourier_transform_stretched(processed_predict_files, sr, meanDuration,feature_size)
#predict_features=feature_exrtaction.extract_mfccs(processed_predict_files, sr, meanDuration)
#predict_features=feature_exrtaction.extract_fourier_peaks(processed_predict_files,sr,meanDuration)

print("Predictions")
# Predict based on trained model
predict.predict(model_path, predict_features, processed_predict_files,"conv")