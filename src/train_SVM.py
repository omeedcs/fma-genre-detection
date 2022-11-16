from utils.dataset import AudioDataset
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from utils.parsers import training_parser


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

preprocessing_dict = {
        "sampling": args.sampling,
        "padding_length": args.padding_length,
        "truncation_len" : args.truncation_length,
        "convert_one_channel": args.convert_one_channel
    }
dataset = AudioDataset(meta_data_path = "data/fma_metadata", audio_folder_path = args.audio_folder_path, preprocessing_dict = preprocessing_dict, debug = args.debug)
x, y = dataset.audio_tensors, dataset.genres_factorized[0]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=137)

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