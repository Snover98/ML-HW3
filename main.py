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

    # normal problem
    # SVC
    svc_normal = SVC(probability=True, class_weight='balanced')
    svc_grid_normal = RandomizedSearchCV(svc_normal, svc_params, scoring=evaluate_voters_division, cv=3)
    svc_grid_normal.fit(train[features], train['Vote'])

    # Random Forest
    rand_forest_normal = RandomForestClassifier(class_weight='balanced')
    random_forest_normal = RandomizedSearchCV(rand_forest_normal, random_forest_params,
                                              scoring=evaluate_voters_division, cv=3)
    random_forest_normal.fit(train[features], train['Vote'])

    # elections results
    # SVC
    svc_elections_results = ElectionsResultsWrapper(SVC(probability=True, class_weight='balanced'))
    svc_grid_elections_results = RandomizedSearchCV(svc_elections_results, wrapper_params(svc_params),
                                                    scoring=evaluate_election_winner, cv=3)
    svc_grid_elections_results.fit(train[features], train['Vote'])

    # Random Forest
    rand_forest_elections_results = ElectionsResultsWrapper(RandomForestClassifier(class_weight='balanced'))
    rand_forest_grid_elections_results = RandomizedSearchCV(rand_forest_elections_results,
                                                            wrapper_params(random_forest_params),
                                                            scoring=evaluate_election_winner, cv=3)
    rand_forest_grid_elections_results.fit(train[features], train['Vote'])

    # likely voters
    threshold_params = {'threshold': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]}
    # SVC
    svc_likely_voters = LikelyVotersWrapper(SVC(probability=True, class_weight='balanced'), threshold=0.5)
    svc_grid_likely_voters = RandomizedSearchCV(svc_likely_voters, wrapper_params(svc_params, threshold_params),
                                                scoring=evaluate_party_voters, cv=3)
    svc_grid_likely_voters.fit(train[features], train['Vote'])

    # Random Forest
    rand_forset_likely_voters = LikelyVotersWrapper(RandomForestClassifier(class_weight='balanced'), 0.5)
    rand_forset_grid_likely_voters = RandomizedSearchCV(rand_forset_likely_voters,
                                                        wrapper_params(random_forest_params, threshold_params),
                                                        scoring=evaluate_party_voters, cv=3)
    rand_forset_grid_likely_voters.fit(train[features], train['Vote'])

    pass
