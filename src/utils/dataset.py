import os
import torchaudio
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
# import h5py


class AudioDataset(Dataset):

    def __init__(self, meta_data_path, audio_folder_path):
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
                    audio_files.append(data_waveform)
                    genres.append(genre)
        self.audio_files = audio_files
        self.genres = genres
        # build pairs of audio file to genre
        return self.audio_files, self.genres

    def __len__(self):
        assert len(self.audio_files) == len(self.genres)
        return len(self.audio_files)
        raise NotImplementedError()
        # return len(self.cat_embeddings)
    
    def input_size(self):
        raise NotImplementedError()
        # return self.cat_embeddings.shape[1]

    def __getitem__(self, idx):
        return self.audio_files[idx], self.genres[idx]
        raise NotImplementedError()