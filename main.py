from data_preperation import prepare_data
import numpy as np
from wrappers import *
from model_selection import *
from scipy.stats import uniform, expon
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class LogUniform:
    def __init__(self, low=0.0, high=1.0, size=None, base=10):
        self.low = low
        self.high = high
        self.size = size
        self.base = base

    def rvs(self, random_state):
        if random_state is None:
            state = np.random
        elif isinstance(random_state, np.random.RandomState):
            state = random_state
        else:
            state = np.random.RandomState(random_state)

        return np.power(self.base, state.uniform(self.low, self.high, self.size))


def print_best_hyper_params(models, problem: str):
    print('============================================')
    print(f'The best hyper-parameters for the {problem} problem are:')
    print('')
    for model in models:
        print(model)
        print('')
    print('')


def print_best_model(model, problem:str):
    print(f'The best model for the {problem} problem is:')
    print(model)
    print('')


def main():
    # train, valid, test = prepare_data()

    train = pd.read_csv('train_processed.csv')
    valid = pd.read_csv('valid_processed.csv')
    test = pd.read_csv('test_processed.csv')

    features = list(set(train.columns.to_numpy().tolist()).difference({'Vote'}))

    random_forest_params = {
        'n_estimators': [10, 20, 50, 80, 100, 120],
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 10, 20, 50],
        'min_samples_leaf': uniform(0, 0.5),
        'max_features': ['auto', 'sqrt', 'log2', None]
    }

    svc_params = {
        'C': LogUniform(low=-4.0, high=3.0),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [3, 4, 5],
        'gamma': ['scale', 'auto'],
        'tol': LogUniform(low=-10, high=0),
        'coef0': [0.0, 1.0]
    }

    estimators = [SVC(probability=True, class_weight='balanced'), RandomForestClassifier(class_weight='balanced')]
    params = [svc_params, random_forest_params]

    # normal problem
    problem = 'voter classification'
    print(f'started {problem}')
    best_normal_estimators = choose_hyper_params(estimators, params, evaluate_voters_division, train, 'Vote')
    print_best_hyper_params(best_normal_estimators, problem)
    best_normal = choose_best_model(best_normal_estimators, train, valid, evaluate_voters_division)
    print_best_model(best_normal, problem)

    # elections results
    problem = 'election results'
    print(f'started {problem}')
    best_election_res_estimators = choose_hyper_params(estimators, params, evaluate_election_winner, train, 'Vote',
                                                       wrapper=ElectionsResultsWrapper)
    print_best_hyper_params(best_election_res_estimators, problem)
    best_election_res = choose_best_model(best_election_res_estimators, train, valid, evaluate_election_winner)
    print_best_model(best_election_res, problem)

    # likely voters
    problem = 'likely voters'
    print(f'started {problem}')
    threshold_params = {'threshold': uniform(0.5, 0.5)}
    best_likely_voters_estimators = choose_hyper_params(estimators, params, evaluate_party_voters, train, 'Vote',
                                                        wrapper=LikelyVotersWrapper, to_add=threshold_params)
    print_best_hyper_params(best_likely_voters_estimators, problem)
    best_likely_voters_model = choose_best_model(best_likely_voters_estimators, train, valid, evaluate_party_voters)
    print_best_model(best_likely_voters_model, problem)


if __name__ == '__main__':
    main()
