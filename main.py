from data_preperation import prepare_data
import numpy as np
from wrappers import *
from model_selection import *
from scipy.stats import uniform
import scipy


class LogUniform:
    def __init__(self, low=0, high=1, size=None, base=10):
        self.low = low
        self.high = high
        self.size = size
        self.base = base

    def rvs(self):
        return np.power(self.base, np.random.uniform(self.low, self.high, self.size))


def main():
    train, valid, test = prepare_data()

    random_forest_params = {
        'n_estimators': [10, 20, 50, 80, 100, 120],
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 3, 4, 5, 10, 20],
        'min_samples_leaf': uniform(),
        'max_features ': ['auto', 'sqrt', 'log2', None],
        'class_weight': ['balanced']
    }

    svm_params = {
        'C': scipy.stats.expon(scale=100),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        'degree': [3, 4, 5],
        'gamma': ['scale', 'auto'],
        'probability': [True],
        'tol': LogUniform(low=-10, high=0),
        'class_weight': ['balanced'],
        'max_iter': [1, 2, 5, 10]
    }





    pass