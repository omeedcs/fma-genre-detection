from utils.paths import *
import argparse

CNNTrainingParser = argparse.ArgumentParser(description = "Process arguments for training of the category dimensional reduction model")
CNNTrainingParser.add_argument("--batch_size", default = 8, help = "Number of samples per batch")
CNNTrainingParser.add_argument("--device", default = "CPU", help = "Device to train on (enabling GPU support")
CNNTrainingParser.add_argument("--num_epochs", default = 5, help = "Number of epochs to train")
CNNTrainingParser.add_argument("--model_name", default = "M5", help = "allows several models to be selected")
CNNTrainingParser.add_argument("--audio_folder_path", default = DATA_PATH / "fma_small", help = "File path to audio data used")
CNNTrainingParser.add_argument("--sampling", default = None, help = "Allows for resampling music waveforms to different sizes")
CNNTrainingParser.add_argument("--padding_length", default = None, help = "Allows to right pad samples too small in length")
CNNTrainingParser.add_argument("--truncation_length", default = 1300000, help = "Specifies max length of a sample")
CNNTrainingParser.add_argument("--convert_one_channel", default = True, help = "Boolean variable to decide if to convert waveform to one channel")
CNNTrainingParser.add_argument("--load_dataset_path", default = None, help = "Path to load compressed dataset to save time")
CNNTrainingParser.add_argument("--debug", default = False, help = "Allows for easy debugging")
CNNTrainingParser.add_argument("--datatype", default = "torch", help = "Allows for easy debugging")