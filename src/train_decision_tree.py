from utils.dataset import AudioDataset

args = CNNTrainingParser.parse_args()


dataset = AudioDataset(meta_data_path = "data/fma_metadata", audio_folder_path = args.audio_folder_path, preprocessing_dict = preprocessing_dict, debug = args.debug)