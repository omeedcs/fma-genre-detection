import os
import torchaudio
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from utils.dataprocessing import * 

# import h5py


class AudioDataset(Dataset):

    def __init__(self, meta_data_path, audio_folder_path, preprocessing_dict = {}):
        track_csv = pd.read_csv(meta_data_path+"/tracks.csv", header=1).iloc[1: , :]
        genres_csv = track_csv["genre_top"]
        audio_files = [] # TODO: make numpy array
        genres = [] # TODO: make numpy array


        for subdir, dirs, files in os.walk(audio_folder_path):
            for filename in files:
                if filename.endswith(('.mp3')):
                    track_id = eval(filename.rstrip(".mp3")) 
                    track_csv_index = track_csv.index[track_csv["Unnamed: 0"] == track_id].tolist()
                    assert len(track_csv_index) == 1
                    genre = genres_csv.iloc[track_csv_index[0]]
                    #print os.path.join(subdir, file)
                    filepath = subdir + os.sep + filename
                    data_waveform, rate_of_sample = torchaudio.load(filepath)
                    data_waveform = self.apply_preproccess(data_waveform, preprocessing_dict)
                    audio_files.append(data_waveform)
                    genres.append(genre)
        audio_files= np.concatenate(audio_files)
        genres= np.concatenate(genres)

        self.audio_files = audio_files
        self.genres = genres
        # build pairs of audio file to genre
        return self.audio_files, self.genres

    def __len__(self):
        assert len(self.audio_files) == len(self.genres)
        return len(self.audio_files)
        # return len(self.cat_embeddings)
    
    def apply_preproccess(self, waveform, proccessing_dict):
        sampling =  proccessing_dict["sampling"]
        assert type(sampling) == dict
        padding_length = proccessing_dict["padding_length"]
        truncation_len = proccessing_dict["truncation_len"]
        convert_one_channel = proccessing_dict["convert_one_channel"]

        if not (padding_length!=None and truncation_len!=None):
            raise ValueError("Invalid processing parameters. One should not pad and truncate the same sample.")
        
        if sampling != None:
            orig_freq, new_freq = sampling["orig_freq"], sampling["new_freq"]
            waveform = resample(waveform, orig_freq, new_freq)

        # Not necessary for our project
        if padding_length != None:
            raise NotImplementedError()

        if truncation_len != None:
            waveform = truncate_sample(waveform, truncation_len)

        if convert_one_channel != None:
            waveform = convert_one_channel(waveform)
        return waveform
    # def input_size(self):
    #     raise NotImplementedError()
    #     # return self.cat_embeddings.shape[1]

    def __getitem__(self, idx):
        return self.audio_files[idx], self.genres[idx]