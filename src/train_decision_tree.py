from utils.dataset import AudioDataset
from sklearn import tree
from sklearn.model_selection import train_test_split

args = CNNTrainingParser.parse_args()


dataset = AudioDataset(meta_data_path = "data/fma_metadata", audio_folder_path = args.audio_folder_path, preprocessing_dict = preprocessing_dict, debug = args.debug)


x_train, x_test, y_train, y_test = train_test_split(dataset, genres, test_size=0.2, random_state=137)

decision_tree = tree.DecisionTreeClassifier(criterion='entropy')
decision_tree = decision_tree.fit(x_train, y_train)