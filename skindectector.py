import matplotlib
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
        split = int(0.75 * len(self._dataset))

        # get the training set
        self._training_set, test = self._dataset[1:split, :], self._dataset[split:, :]
        self._test_set = test[:, :-1]
        self._test_set_labels = test[:, [-1]]

        self.prob, self.labels = self.train(self._training_set, self._test_set, self._test_set_labels, 2)

    def train(self, training_set, test, labels, K):
        """Takes some skin/no skin training examples and returns the params"""
        probs = classifier.gen_classifier(training_set, test,  K)
        return probs, labels


skindetector = SkinDetector("./data/Skin_NonSkin.txt")
for i in range(len(skindetector.prob)):
    if skindetector.prob[i][0] > 0.6:
        skindetector.prob[i][0] = 1
    else:
        skindetector.prob[i][0] = 0

counter = 0
for i in range(len(skindetector.prob)):
    if skindetector.prob[i][0] == 1 and skindetector.labels[i][0] == 0:
        counter += 1
    elif skindetector.prob[i][0] == 0 and skindetector.labels[i][0] == 1:
        counter += 1

accuracy = (1 - (counter/len(skindetector.prob))) * 100
print(round(accuracy, 3))
