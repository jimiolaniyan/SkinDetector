from __future__ import division
from scipy.stats import multivariate_normal
import numpy as np


def train_classifier(x_train, y_train, K):
    """
    Train a basic generative classifier to get a probability distribution
    over the data, parameterised by mean and covariance
    :param x_train: training set data (n x m matrix)
    :param y_train: labels (n x 1 column vector)
    :param K: Number classes in the data set
    :return: prior probabilities, mean and covariance
    """

    # Ensure we have a numpy array
    if type(x_train) != np.ndarray:
        x_train = np.array(x_train)

    if type(y_train) != np.ndarray:
        y_train = np.array(y_train)

    mean = []
    covariance = []

    # Use to get prior probabilities over the world state
    lammda = np.zeros((K, 1))

    # the counts of each class
    counts = np.bincount(np.squeeze(y_train))

    # Get params (theta) by class
    for k in range(0, K):
        # Get the numerator for the mean of the class k
        k_mean_sum = mean_sum(k, x_train, y_train)

        # Get the mean of the class k (Dx1 matrix)
        k_mean = k_mean_sum / counts[k]
        mean.append(k_mean)

        # Get the numerator for the covariance of the class k
        k_var_sum = var_sum(k, x_train, k_mean, y_train)

        # Get the covariance of the class k (DxD matrix)
        k_var = k_var_sum / counts[k]

        # Prior probability of class k
        lammda[k] = counts[k] / x_train.shape[0]
        covariance.append(k_var)

    return lammda, mean, covariance


# A basic generative classifier
def test_classifier(prior, mean, covariance, x_test, K):
    """
    Test the generative classifier
    :param prior: Prior probability distribution for each class
    :param mean: K x 1 vector with the mean of the kth class in the kth position
    :param covariance: K x 1 vector with the covariance of the kth class in the kth position
    :param x_test: test set
    :param K: Number classes in the data set
    :return: Posterior probability
    """
    # Compute likelihoods for each class for data points in test set
    likelihood = np.zeros((x_test.shape[0], K))
    for k in range(0, K):
        cov_mat = covariance[k] * np.eye(covariance[0].shape[0])
        l_val = multivariate_normal.pdf(x_test, cov=cov_mat, mean=mean[k])
        likelihood[:, k] = l_val

    # Classify new data point using Bayes rule
    # Get the denominator
    denom = 1 / np.dot(likelihood, prior)

    # Calculate the posterior probability of each data point in test set
    posterior_prob = likelihood * prior.T * denom
    return posterior_prob[:, [1]]


# helper method to get the sum by class for mean calculation
def mean_sum(k, x, labels):
    mu_sum = 0
    for i in range(0, len(x)):
        if labels[i][0] == k:
            mu_sum += x[i]
    return mu_sum


# helper method to get the sum by class for covariance calculation
def var_sum(k, x, mu, labels):
    cov = 0
    for i in range(0, len(x)):
        if labels[i][0] == k:
            mu_diff = x[i] - mu
            mu_diff_col = np.array([mu_diff])
            diff_by_trans = mu_diff_col.T * mu_diff
            cov += diff_by_trans
    return cov
