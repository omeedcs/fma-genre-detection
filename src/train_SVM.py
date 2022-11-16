from sklearn import svm


dataset = AudioDataset(meta_data_path = "data/fma_metadata", audio_folder_path = args.audio_folder_path, preprocessing_dict = preprocessing_dict, debug = args.debug)

model = svm.SVC(kernel='linear')
model.fit(x_train, y_train)