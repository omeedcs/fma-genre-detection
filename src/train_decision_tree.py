from utils.dataset import AudioDataset
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from utils.parsers import training_parser
import time

args = training_parser.parse_args()
args.batch_size = 4 # TODO: delete later
args.device = "cpu"
args.num_epochs = 5
args.model_name = "M5"
args.audio_folder_path = "data/fma_small"
args.sampling_freq = 10000
args.padding_length = None
args.truncation_length = 10000 # TODO_modify
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
        "sampling_freq": args.sampling_freq,
        "padding_length": args.padding_length,
        "truncation_len" : args.truncation_length,
        "convert_one_channel": args.convert_one_channel
    }
start = time.time()
USE_GRID_SEARCH = False
dataset = AudioDataset(meta_data_path = "data/fma_metadata", audio_folder_path = args.audio_folder_path, preprocessing_dict = preprocessing_dict, debug = args.debug)


x_train, x_test, y_train, y_test = train_test_split(dataset.audio_tensors, dataset.genres_factorized[0], test_size=0.2, random_state=24)
decision_tree = tree.DecisionTreeClassifier(criterion='entropy')
if USE_GRID_SEARCH:
    possible_values = {'max_depth': [5,10,15,20], 'min_samples_leaf': [5,10,15,20]}
    best_tree = GridSearchCV(decision_tree, param_grid=possible_values, cv=5, scoring='accuracy')
    best_tree.fit(x_train, y_train)
    y_pred = best_tree.predict(x_test)
else:
    decision_tree.fit(x_train, y_train)
    y_pred = decision_tree.predict(x_test)

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

stop = time.time()
print("The time of the run:", stop - start)