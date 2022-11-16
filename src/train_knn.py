from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from utils.dataset import AudioDataset
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import numpy as np

args = training_parser.parse_args()
args.batch_size = 4 # TODO: delete later
args.device = "cpu"
args.num_epochs = 5
args.model_name = "M5"
args.audio_folder_path = "data/fma_small"
args.sampling = None # {"orig_freq": None, "new_freq": None}
args.padding_length = None
args.truncation_length = 1300000
args.convert_one_channel = True
args.load_dataset_path = None # or logs/datasets/dataset_fma_small_one_channel
args.debug = True  # TODO delete
args.desired_dataset_name = "dataset_fma_small_one_channel_np"
args.datatype = "np"
if args.audio_folder_path == "data/fma_small":
    num_genres = 8
else:
    raise NotImplementedError()

preprocessing_dict = {
        "sampling": args.sampling,
        "padding_length": args.padding_length,
        "truncation_len" : args.truncation_length,
        "convert_one_channel": args.convert_one_channel
    }


dataset = AudioDataset(meta_data_path = "data/fma_metadata", audio_folder_path = args.audio_folder_path, preprocessing_dict = preprocessing_dict, debug = args.debug)
x_train, x_test, y_train, y_test = train_test_split(dataset.audio_tensors, dataset.genres_factorized[0], test_size=0.2, random_state=24)

# train knn model
# we use 8 neighbors because there are 8 genres in dataset
knn = KNeighborsClassifier(n_neighbors=8)

# train model with cv of 5
# we use 5-fold cross validation because we have quite a small dataset.
cv_scores = cross_val_score(knn, x_train, y_train, cv=5)

# we find the mean of the 5 scores because we want to find the average accuracy of the model
cv_scores_mean = np.mean(cv_scores)

# we print the mean of the 5 scores, aka the average accuracy of the model
print(cv_scores , "\n\n""mean =" ,"{:.2f}".format(cv_scores_mean))

predictions = knn.predict(x_test)
accuracy_score = knn.score(x_test, y_test)
print("Accuracy score = ""{:.2f}".format(accuracy_score))

# confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, predictions)
print(confusion_matrix)

# classification report
classification_report = metrics.classification_report(y_test, predictions)
print(classification_report)

# precision score
precision_score = metrics.precision_score(y_test, predictions, average='weighted')
print("Precision score = ""{:.2f}".format(precision_score))

# recall score
recall_score = metrics.recall_score(y_test, predictions, average='weighted')
print("Recall score = ""{:.2f}".format(recall_score))