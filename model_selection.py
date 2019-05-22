import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from numpy.linalg import norm


def evaluate_voters_division(estimator, X, y_true) -> float:
    y_pred = estimator.predict(X)
    return accuracy_score(y_true, y_pred)


def evaluate_election_winner(estimator, X, y_true) -> float:
    lb = LabelBinarizer()
    lb.fit(y_true)

    hist_true = np.sum(lb.transform(y_true), axis=0)
    hist_pred = estimator.predict_proba(X)

    return -norm(hist_true-hist_pred)


def evaluate_party_voters(estimator, X, y_true, party):
    indices_true = y_true.index[y_true == party]
    indices_pred = estimator.predict(X)

    true_pos_indices = list(set(indices_true).intersection(set(indices_pred)))

    return np.sqrt((len(true_pos_indices) ** 2) / (len(indices_true) * len(indices_pred)))


def evaluate_likely_voters(estimator, X, y_true):
    """

    :param estimator: estimator
    :param y_true: should be valid['Vote'] or something like that
    :param X: should be valid[features] or something like that
    :return: average score for each party
    """
    parties = y_true.unique()

    return np.mean([evaluate_party_voters(estimator, y_true, X, party) for party in parties])


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
    score = -np.inf

    df_targets, df_features = target_features_split(df, 'Vote')

    for train_indices, test_indices in kf.split(df_features, df_targets):
        train_targets, train_features = df_targets[train_indices], df_features[train_indices]
        test_targets, test_features = df_targets[test_indices], df_features[test_indices]

        model.fit(train_features, train_targets)
        score += eval_func(model, test_targets, test_features)

    return score / num_folds


def choose_best_model(models, train: pd.DataFrame, valid: pd.DataFrame, eval_func):
    best_score = -np.inf
    best_model = None

    train_targets, train_features = target_features_split(train, 'Vote')
    valid_targets, valid_features = target_features_split(valid, 'Vote')
    
    for model in models:
        model.fit(train_features, train_targets)
        score = eval_func(model, valid_targets, valid_features)

        if score > best_score:
            best_score = score
            best_model = model

    return best_model
