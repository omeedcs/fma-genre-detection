from utils.dataset import AudioDataset
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

data, genres = AudioDataset(meta_data_path = "data/fma_metadata", audio_folder_path = args.audio_folder_path, preprocessing_dict = preprocessing_dict, debug = args.debug)
x_train, x_test, y_train, y_test = train_test_split(data, genres, test_size=0.2, random_state=137)

# train svm model
model = svm.SVC()
model.fit(x_train, y_train)

# make predictions on test set
y_pred = model.predict(x_test)

# test
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# train kernel svm model
model = svm.SVC(kernel='poly', degree=2,)
model.fit(x_train, y_train)

# make predictions on test set
y_pred = model.predict(x_test)

# test
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Out of the 4 types of available kernels for SVMs: Linear, Polynomial, Radial Basis Function, and Sigmoid; the Polynomial appears to be the most useful for this type of data
# as it is the best for classifing natural language. 
# https://www.kaggle.com/code/prashant111/svm-classifier-tutorial/notebook
# https://link.springer.com/article/10.1007/s42452-020-03870-0