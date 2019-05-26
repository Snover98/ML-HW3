import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from numpy.linalg import norm


def evaluate_voters_division(estimator, X, y_true) -> float:
    y_pred = estimator.predict(X)
    return balanced_accuracy_score(y_true, y_pred)


# def evaluate_election_winner(estimator, X, y_true) -> float:
#     y_pred = estimator.predict(X)
#     return accuracy_score(y_true, y_pred)


def evaluate_election_res(estimator, X, y_true) -> float:
    hist_true = y_true.value_counts()
    hist_pred = estimator.predict_proba(X)

    return -norm(hist_true - hist_pred)


def evaluate_party_voters(indices_pred, y_true: pd.Series):
    y_pred = y_true.copy()
    y_pred[indices_pred] = True
    y_pred[y_pred.index.difference(indices_pred)] = False

    return f1_score(y_true, y_pred)


def evaluate_likely_voters(estimator, X, y_true: pd.Series):
    """

    :param estimator: estimator
    :param y_true: should be valid['Vote'] or something like that
    :param X: should be valid[features] or something like that
    :return: average score for each party
    """
    parties = estimator.targets
    y_pred = estimator.predict(X)

    return np.mean([evaluate_party_voters(y_pred[party], y_true == party) for party in parties])


def upsample(df: pd.DataFrame, target: str) -> pd.DataFrame:
    targets = df[target]
    classes = targets.unique()

    num_appearances = {target_class: targets[targets == target_class].size for target_class in classes}
    most_common = max(num_appearances.iterkeys(), key=(lambda key: num_appearances[key]))

    minority_upsampled = [resample(df[targets == target_class], replace=True, n_samples=num_appearances[most_common])
                          for target_class in classes if target_class != most_common]

    df_upsampled = pd.concat([df[targets == most_common]] + minority_upsampled)

    return df_upsampled


def target_features_split(df: pd.DataFrame, target: str):
    features = list(set(df.columns.to_numpy().tolist()).difference({target}))
    return df[target], df[features]


def cross_valid(model, df: pd.DataFrame, num_folds: int, eval_func):
    kf = StratifiedKFold(n_splits=num_folds)
    score = 0

    df_targets, df_features = target_features_split(df, 'Vote')

    for train_indices, test_indices in kf.split(df_features, df_targets):
        train_targets, train_features = df_targets[train_indices], df_features.iloc[train_indices]
        test_targets, test_features = df_targets[test_indices], df_features.iloc[test_indices]

        model.fit(train_features, train_targets)
        score += eval_func(model, test_features, test_targets)

    return score / num_folds


def choose_best_model(models, train: pd.DataFrame, valid: pd.DataFrame, eval_func, verbose: bool = False):
    best_score = -np.inf
    best_model = None

    train_targets, train_features = target_features_split(train, 'Vote')
    valid_targets, valid_features = target_features_split(valid, 'Vote')

    for model in models:
        model.fit(train_features, train_targets)
        score = eval_func(model, valid_features, valid_targets)

        if verbose:
            model_name: str = model.__repr__().split('(')[0]
            if model_name.endswith('Wrapper'):
                model_name = model.model.__repr__().split('(')[0]

            print(f'Model {model_name} has a score of {score}')

        if score > best_score:
            best_score = score
            best_model = model

    return best_model


def wrapper_params(params: dict):
    return {'model__' + key: value for key, value in params.items()}


def choose_hyper_params(models, params_ranges, eval_func, df, target='Vote', num_folds=3, wrapper=None,
                        random_state=None, n_iter=10):
    used_models = models
    if wrapper is not None:
        used_models = [wrapper(model) for model in models]

    y, X = target_features_split(df, target)

    best_models = []
    for model, params in zip(used_models, params_ranges):
        print(f'doing model #{len(best_models) + 1}')
        used_params = params
        if wrapper is not None:
            used_params = wrapper_params(params)

        grid = RandomizedSearchCV(model, used_params, scoring=eval_func, cv=num_folds, random_state=random_state,
                                  n_iter=n_iter, n_jobs=-1)
        grid.fit(X, y)
        best_models.append(grid.best_estimator_)

    return best_models
