import os
import torch
# import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
# import h5py


class AudioDataset(Dataset):

    def __init__(self, meta_data_path, audio_folder_path):
       raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()
        # return len(self.cat_embeddings)
    
    def input_size(self):
        raise NotImplementedError()
        # return self.cat_embeddings.shape[1]

    def __getitem__(self, idx):
        raise NotImplementedError()