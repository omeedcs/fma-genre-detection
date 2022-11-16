from utils.paths import *
import argparse

training_parser = argparse.ArgumentParser(description = "Process arguments for training of the category dimensional reduction model")
training_parser.add_argument("--batch_size", default = 8, help = "Number of samples per batch")
training_parser.add_argument("--device", default = "CPU", help = "Device to train on (enabling GPU support")
training_parser.add_argument("--num_epochs", default = 5, help = "Number of epochs to train")
training_parser.add_argument("--model_name", default = "M5", help = "allows several models to be selected")
training_parser.add_argument("--audio_folder_path", default = DATA_PATH / "fma_small", help = "File path to audio data used")
training_parser.add_argument("--sampling_freq", default = None, help = "Allows for resampling music waveforms to different sizes")
training_parser.add_argument("--padding_length", default = None, help = "Allows to right pad samples too small in length")
training_parser.add_argument("--truncation_length", default = 1300000, help = "Specifies max length of a sample")
training_parser.add_argument("--convert_one_channel", default = True, help = "Boolean variable to decide if to convert waveform to one channel")
training_parser.add_argument("--load_dataset_path", default = None, help = "Path to load compressed dataset to save time")
training_parser.add_argument("--debug", default = False, help = "Allows for easy debugging")
training_parser.add_argument("--datatype", default = "torch", help = "Allows for easy debugging")