import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import KFold
from sklearn.utils import resample
from sklearn.metrics import accuracy_score


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


def cross_valid(model, df: pd.DataFrame, num_folds: int):
    kf = KFold(n_splits=num_folds)
    acc = 0
    for train_indexes, test_indexes in kf.split(df):
        train_set = df[train_indexes]
        test_set = df[test_indexes]
        train_targets, train_features = target_features_split(train_set, 'Vote')
        test_targets, test_features = target_features_split(test_set, 'Vote')
        model.fit(train_features, train_targets)
        pred = model.predict(test_features)
        acc += accuracy_score(test_targets, pred)
    return acc / num_folds


def choose_best_model(models, train: pd.DataFrame, valid: pd.DataFrame):
    best_acc = 0
    best_model = None

    train_targets, train_features = target_features_split(train, 'Vote')
    valid_targets, valid_features = target_features_split(valid, 'Vote')
    
    for model in models:
        model.fit(train_features, train_targets)
        pred = model.predict(valid_features)
        acc = accuracy_score(valid_targets, pred)
        if acc > best_acc:
            best_acc = acc
            best_model = model
    return best_model
