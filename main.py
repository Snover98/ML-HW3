from data_preperation import prepare_data
import numpy as np
from wrappers import *
from model_selection import *
from scipy.stats import uniform, expon
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint


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


class RandIntMult:
    def __init__(self, low=0.0, high=1.0, mult=1, size=None):
        self.low = low
        self.high = high
        self.size = size
        self.mult = mult

    def rvs(self, random_state):
        if random_state is None:
            state = np.random
        elif isinstance(random_state, np.random.RandomState):
            state = random_state
        else:
            state = np.random.RandomState(random_state)

        return np.around(state.uniform(low=self.low, high=self.high, size=self.size) * self.mult).astype(int)


def print_best_hyper_params(models, problem: str):
    print('============================================')
    print(f'The best hyper-parameters for the {problem} problem are:')
    print('')
    for model in models:
        print(model)
        print('')
    print('')


def print_best_model(model, problem: str):
    print(f'The best model for the {problem} problem is:')
    print(model)
    print('')


def main():
    # train, valid, test = prepare_data()

    train = pd.read_csv('train_processed.csv')
    valid = pd.read_csv('valid_processed.csv')
    test = pd.read_csv('test_processed.csv')

    features = list(set(train.columns.to_numpy().tolist()).difference({'Vote'}))

    seed = np.random.randint(2 ** 31)
    print(f'seed is {seed}')

    n_iter = 10

    random_forest_params = {
        'n_estimators': RandIntMult(low=0.5, high=20.0, mult=100),
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2', None],
        'bootstrap': [True, False],
        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
    }

    svc_params = {
        'C': LogUniform(low=-5.0, high=4.0),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [3, 4, 5],
        'gamma': ['scale', 'auto'],
        'tol': LogUniform(low=-10, high=0),
        'coef0': [0.0, 1.0]
    }

    estimators = [SVC(probability=True, class_weight='balanced'),
                  RandomForestClassifier(class_weight='balanced', n_jobs=-1)]
    params = [svc_params, random_forest_params]

    # normal problem
    problem = 'voter classification'
    print(f'started {problem}')
    best_normal_estimators = choose_hyper_params(estimators, params, evaluate_voters_division, train, 'Vote',
                                                 random_state=seed, n_iter=n_iter)
    print_best_hyper_params(best_normal_estimators, problem)
    best_normal = choose_best_model(best_normal_estimators, train, valid, evaluate_voters_division, verbose=True)
    print_best_model(best_normal, problem)

    # elections winner
    problem = 'election winner'
    print(f'started {problem}')
    best_election_win_estimators = choose_hyper_params(estimators, params, evaluate_election_winner, train, 'Vote',
                                                       wrapper=ElectionsWinnerWrapper, random_state=seed, n_iter=n_iter)
    print_best_hyper_params(best_election_win_estimators, problem)
    best_election_win = choose_best_model(best_election_win_estimators, train, valid, evaluate_election_winner,
                                          verbose=True)
    print_best_model(best_election_win, problem)

    # elections results
    problem = 'election results'
    print(f'started {problem}')
    best_election_res_estimators = choose_hyper_params(estimators, params, evaluate_election_res, train, 'Vote',
                                                       wrapper=ElectionsResultsWrapper, random_state=seed,
                                                       n_iter=n_iter)
    print_best_hyper_params(best_election_res_estimators, problem)
    best_election_res = choose_best_model(best_election_res_estimators, train, valid, evaluate_election_res,
                                          verbose=True)
    print_best_model(best_election_res, problem)

    # likely voters
    problem = 'likely voters'
    print(f'started {problem}')
    best_likely_voters_estimators = choose_hyper_params(estimators, params, evaluate_likely_voters, train, 'Vote',
                                                        wrapper=LikelyVotersWrapper, random_state=seed, n_iter=n_iter)
    print_best_hyper_params(best_likely_voters_estimators, problem)
    best_likely_voters_model = choose_best_model(best_likely_voters_estimators, train, valid, evaluate_likely_voters,
                                                 verbose=True)
    print_best_model(best_likely_voters_model, problem)

    # data for the final classifier
    print('============================================')
    non_test_data = pd.concat((train, valid))
    best_normal.fit(non_test_data[features], non_test_data['Vote'])
    test_pred = pd.Series(best_normal.predict(test[features]), index=test.index)
    test_true = test['Vote']
    conf_matrix = confusion_matrix(test_true, test_pred, non_test_data['Vote'].unique())
    print('The confusion matrix is:')
    print(conf_matrix)
    print('')
    test_pred.to_csv('test_predictions.csv', index=False)

    # predict elections winner
    print('============================================')
    best_election_win.fit(non_test_data[features], non_test_data['Vote'])
    pred_election_winner = best_election_win.predict(test[features])
    true_election_winner = test['Vote'].value_counts().idxmax()
    print(f'The predicted elections winner is {pred_election_winner} and the actual winner is {true_election_winner}')
    print('')

    # predict elections results
    print('============================================')
    best_election_res.fit(non_test_data[features], non_test_data['Vote'])
    pred_percantages = best_election_res.predict(train[features]).value_counts() / len(test.index) * 100
    true_percantages = test['Vote'].value_counts() / len(test.index) * 100
    print('The predicted distribution of votes across the parties is:')
    print(pred_percantages)
    print('The true distribution of votes across the parties is:')
    print(true_percantages)
    print('')

    # predict likely voters
    print('============================================')
    best_likely_voters_estimators.fit(non_test_data[features], non_test_data['Vote'])
    pred_likely_voters = best_likely_voters_model.predict(test[features])
    actual_voters = {party: test['Vote'].index[test['Vote'] == party] for party in non_test_data['Vote'].unique()}
    print('Predicted likely voter indices per party:')
    pprint(pred_likely_voters)
    print('Actual voter indices per party:')
    pprint(actual_voters)
    print('')


if __name__ == '__main__':
    main()
