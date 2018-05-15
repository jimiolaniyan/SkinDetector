import pandas as pd
from skin_detector import classifier
import numpy as np

class SkinDetector:

    def __init__(self, path):
        # read the data. file is txt.
        dataset = pd.read_csv(path, delimiter="\t", header=None, names=["R", "G", "B", "y"])

        # replace 2 with zeros
        dataset.y.replace([2, 1], [0, 1], inplace=True)

        # shuffle the data
        dataset = dataset.sample(frac=1).reset_index(drop=True)

        # get a dataset into a numpy array
        self._dataset = dataset.iloc[:, :].values

        # split the data set
        split = int(0.8 * len(self._dataset))

        # get the training set
        training_set, test = self._dataset[1:split, :], self._dataset[split:, :]
        self._x_train = training_set[:, :-1]
        self._y_train = training_set[:, [-1]]
        self._x_test = test[:, :-1]
        self._y_test = test[:, [-1]]

        # Assign the classes (there are only two classes, either skin or non-skin)
        self.classes = 2

    def train(self):
        """Takes some skin/no skin training examples and returns the params"""
        return classifier.train_classifier(self._x_train, self._y_train,  self.classes)

    def test(self, prior, mu, covariance, x_test="default"):
        if x_test == "default":
            x_test = self._x_test
            return classifier.test_classifier(prior, mu, covariance, x_test, self.classes), self._y_test
        else:
            return classifier.test_classifier(prior, mu, covariance, x_test, self.classes)