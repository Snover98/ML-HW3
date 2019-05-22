from data_preperation import prepare_data
import numpy as np
from wrappers import *
from model_selection import *
from scipy.stats import uniform, expon
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class LogUniform:
    def __init__(self, low=0, high=1, size=None, base=10):
        self.low = low
        self.high = high
        self.size = size
        self.base = base

    def rvs(self):
        return np.power(self.base, np.random.uniform(self.low, self.high, self.size))


def wrapper_params(params: dict, to_add: dict = {}):
    new_params = {'model__' + key: item for key, item in params.items()}
    new_params.update(to_add)
    return new_params


def main():
    train, valid, test = prepare_data()
    features = list(set(train.to_numpy().tolist()).difference({'Vote'}))

    random_forest_params = {
        'n_estimators': [10, 20, 50, 80, 100, 120],
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 3, 4, 5, 10, 20],
        'min_samples_leaf': uniform(),
        'max_features ': ['auto', 'sqrt', 'log2', None]
    }

    svc_params = {
        'C': expon(scale=100),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        'degree': [3, 4, 5],
        'gamma': ['scale', 'auto'],
        'tol': LogUniform(low=-10, high=0)
    }

    svc_normal = SVC(probability=True, class_weight='balanced')
    rand_forest_normal = RandomForestClassifier(class_weight='balanced')

    svc_grid_normal = RandomizedSearchCV(svc_normal, svc_params, scoring=evaluate_voters_division, cv=3)
    svc_grid_normal.fit(train[features], train['Vote'])

    random_forest_normal = RandomizedSearchCV(rand_forest_normal, random_forest_params,
                                              scoring=evaluate_voters_division, cv=3)
    random_forest_normal.fit(train[features], train['Vote'])

    pass
