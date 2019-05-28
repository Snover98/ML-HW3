import pandas as pd
import sklearn as sk
from data_preperation import prepare_data
from wrappers import *
from model_selection import *
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
import pickle


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


def model_problem_name(model: sk.base.BaseEstimator, problem: str) -> str:
    model_name: str = get_model_name(model)

    return f'{problem}_{model_name}.pickle'


def save_model_problem_hyper_params(model: sk.base.BaseEstimator, problem: str):
    with open(model_problem_name(model, problem), 'wb') as handle:
        pickle.dump(model.get_params, handle)


def save_problem_hyper_params(models, problem: str):
    for model in models:
        save_model_problem_hyper_params(model, problem)


def load_model_problem_hyper_params(model, problem: str, verbose=False):
    with open(model_problem_name(model, problem), 'wb') as handle:
        params = pickle.load(handle)

    if verbose:
        print(f'For the problem {problem} the best hyper parameters for the estimator {get_model_name(model)} are:')
        print_params = {(key.split('model__')[1] if key.startswith('model') else key): value
                        for key, value in params.items() if key != 'model'}
        print(print_params)

    model.set_params(**params)


def copy_model(model):
    return type(model)(**model.get_params())


def load_problem_hyper_params(models, problem: str, wrapper=None, verbose=False):
    used_models = [copy_model(model) for model in models]
    if wrapper is not None:
        used_models = [wrapper(model) for model in used_models]

    for model in used_models:
        load_model_problem_hyper_params(model, problem, verbose)

    return used_models


def print_best_hyper_params(models, problem: str):
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


def find_best_models(train, valid, search_hyper_params=True, verbose=False):
    seed = np.random.randint(2 ** 31)
    print(f'seed is {seed}')
    print('')

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
    print('============================================')
    problem = 'voter classification'
    print(f'started {problem}')
    if search_hyper_params:
        best_normal_estimators = choose_hyper_params(estimators, params, evaluate_voters_division, train, 'Vote',
                                                     random_state=seed, n_iter=n_iter)
        save_problem_hyper_params(best_normal_estimators, problem)
    else:
        best_normal_estimators = load_problem_hyper_params(estimators, problem, verbose=verbose)

    print_best_hyper_params(best_normal_estimators, problem)
    best_normal = choose_best_model(best_normal_estimators, valid, evaluate_voters_division, verbose=verbose)
    print_best_model(best_normal, problem)

    # elections winner
    print('============================================')
    problem = 'election winner'
    print(f'started {problem}')
    if search_hyper_params:
        best_election_win_estimators = choose_hyper_params(estimators, params, evaluate_election_winner, train, 'Vote',
                                                           wrapper=ElectionsWinnerWrapper, random_state=seed,
                                                           n_iter=n_iter, verbose=verbose)
        save_problem_hyper_params(best_election_win_estimators, problem)
    else:
        best_election_win_estimators = load_problem_hyper_params(estimators, problem, wrapper=ElectionsWinnerWrapper,
                                                                 verbose=verbose)

    print_best_hyper_params(best_election_win_estimators, problem)
    best_election_win = choose_best_model(best_election_win_estimators, valid, evaluate_election_winner,
                                          verbose=verbose)
    print_best_model(best_election_win, problem)

    # elections results
    print('============================================')
    problem = 'election results'
    print(f'started {problem}')
    if search_hyper_params:
        best_election_res_estimators = choose_hyper_params(estimators, params, evaluate_election_res, train, 'Vote',
                                                           wrapper=ElectionsResultsWrapper, random_state=seed,
                                                           n_iter=n_iter, verbose=verbose)
        save_problem_hyper_params(best_election_res_estimators, problem)
    else:
        best_election_res_estimators = load_problem_hyper_params(estimators, problem, wrapper=ElectionsResultsWrapper,
                                                                 verbose=verbose)

    print_best_hyper_params(best_election_res_estimators, problem)
    best_election_res = choose_best_model(best_election_res_estimators, valid, evaluate_election_res, verbose=verbose)
    print_best_model(best_election_res, problem)

    # likely voters
    print('============================================')
    problem = 'likely voters'
    print(f'started {problem}')
    if search_hyper_params:
        best_likely_voters_estimators = choose_hyper_params(estimators, params, evaluate_likely_voters, train, 'Vote',
                                                            wrapper=LikelyVotersWrapper, random_state=seed,
                                                            n_iter=n_iter, verbose=verbose)
        save_problem_hyper_params(best_likely_voters_estimators, problem)
    else:
        best_likely_voters_estimators = load_problem_hyper_params(estimators, problem, wrapper=LikelyVotersWrapper,
                                                                  verbose=verbose)

    print_best_hyper_params(best_likely_voters_estimators, problem)
    best_likely_voters_model = choose_best_model(best_likely_voters_estimators, valid, evaluate_likely_voters,
                                                 verbose=verbose)
    print_best_model(best_likely_voters_model, problem)

    return best_normal, best_election_win, best_election_res, best_likely_voters_model


def use_estimators(best_estimators, train, valid, test):
    best_normal, best_election_win, best_election_res, best_likely_voters_model = best_estimators

    features = list(set(train.columns.to_numpy().tolist()).difference({'Vote'}))

    non_test_data = pd.concat((train, valid))

    # data for the final classifier
    print('============================================')
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
    best_likely_voters_model.fit(non_test_data[features], non_test_data['Vote'])
    pred_likely_voters = best_likely_voters_model.predict(test[features])
    actual_voters = {party: test['Vote'].index[test['Vote'] == party] for party in non_test_data['Vote'].unique()}
    print('Predicted likely voter indices per party:')
    pprint(pred_likely_voters)
    print('Actual voter indices per party:')
    pprint(actual_voters)
    print('')


def main():
    # train, valid, test = prepare_data()

    train = pd.read_csv('train_processed.csv')
    valid = pd.read_csv('valid_processed.csv')
    test = pd.read_csv('test_processed.csv')

    best_models = find_best_models(train, valid)
    use_estimators(best_models, train, valid, test)


if __name__ == '__main__':
    main()
