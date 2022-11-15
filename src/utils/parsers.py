from utils.paths import *
import argparse

CNNTrainingParser = argparse.ArgumentParser(description = "Process arguments for training of the category dimensional reduction model")
CNNTrainingParser.add_argument("--batch_size", "-b", default = 8, help = "Number of samples per batch")
CNNTrainingParser.add_argument("--device", "-p", default = "CPU", help = "Device to train on (enabling GPU support")
CNNTrainingParser.add_argument("--num_epochs", "-p", default = 5, help = "Number of epochs to train")
CNNTrainingParser.add_argument("--model_name", "-p", default = "M5", help = "allows several models to be selected")
CNNTrainingParser.add_argument("--audio_folder_path", "-p", default = DATA_PATH / "fma_small", help = "File path to audio data used")
CNNTrainingParser.add_argument("--sampling", "-p", default = None, help = "Allows for resampling music waveforms to different sizes")
CNNTrainingParser.add_argument("--padding_length", "-p", default = None, help = "Allows to right pad samples too small in length")
CNNTrainingParser.add_argument("--truncation_length", "-p", default = None, help = "Specifies max length of a sample")
CNNTrainingParser.add_argument("--convert_one_channel", "-p", default = True, help = "Boolean variable to decide if to convert waveform to one channel")