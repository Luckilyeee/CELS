from pyts.datasets import fetch_ucr_dataset
from sklearn.preprocessing import LabelEncoder

class BeetleFlyDataset():
    def __init__(self):
        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None

    def load_train_data(self):
        train_data, test_data, train_label, test_label = fetch_ucr_dataset("BeetleFly", use_cache=True, data_home=None, return_X_y=True)
        print("load train data: ", train_data.shape)
        encoder = LabelEncoder()
        self.train_label = encoder.fit_transform(train_label)
        self.train_data = train_data
        return self.train_data, self.train_label

    def load_test_data(self):
        train_data, test_data, train_label, test_label = fetch_ucr_dataset("BeetleFly", use_cache=True, data_home=None, return_X_y=True)
        encoder = LabelEncoder()
        self.test_label = encoder.fit_transform(test_label)
        self.test_data = test_data
        return self.test_data, self.test_label

