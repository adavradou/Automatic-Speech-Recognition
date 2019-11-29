import os
from pathlib import Path

from Deliverable import signal_processing

current_directory = os.getcwd()

# Choose audio dataset
train_audio_path = Path('./free-spoken-digit-dataset-medium') #Audio Path

print("Training Dataset Folder:",train_audio_path)

labels = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

processedAudioFiles,meanDuration= signal_processing.processDataset(labels, train_audio_path)

