from models.M5_Audio_Classifier import *
from utils.dataset import AudioDataset

if __name__ == "__main__":
    dataset = AudioDataset(meta_data_path = "data/fma_metadata", audio_folder_path = "data/fma_small")
    # model = M5(n_input=transformed.shape[0], n_output=len(labels))
    # model.to(device)
    # print(model)
