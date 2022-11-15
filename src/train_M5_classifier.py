from models.M5_Audio_Classifier import *
from utils.dataset import AudioDataset
from torch.utils.data import Dataset, Dataloader
from utils.parsers import AETrainingParser


if __name__ == "__main__":
    args = AETrainingParser.parse_args()
    args.batch_size = 4 # TODO: delete later
    dataset = AudioDataset(meta_data_path = "data/fma_metadata", audio_folder_path = "data/fma_small")
    dataloader = Dataloader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=0)
    # TODO: implement training loop
    # model = M5(n_input=transformed.shape[0], n_output=len(labels))
    # model.to(device)
    # print(model)
