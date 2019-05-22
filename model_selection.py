import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import KFold


def split_lab_exam(df: pd.DataFrame):
    featuers = []
    for col in df:
        if col != "Vote":
            featuers.append(col)
    labels = df["Vote"]
    exampels = df[featuers]
    return labels, exampels


def cross_valid(model, validation_set: pd.DataFrame, num_folds: int):
    kf = KFold(n_splits=num_folds)
    acc = 0
    for train_indexes, test_indexes in kf.split(validation_set):
        train_set = validation_set[train_indexes]
        test_set = validation_set[test_indexes]
        train_labels, train_exampels = split_lab_exam(train_set)
        test_labels, test_exampels = split_lab_exam(test_set)
        model.fit(train_exampels, train_labels)
        pred = model.predict(test_exampels)
        acc += sum(pred == test_labels) / len(test_indexes)
    return acc / num_folds


def choose_best_model(models, df: pd.DataFrame, num_folds: int):
    best_acc = 0
    best_model = models[0]
    for model in models:
        labels, exampels = split_lab_exam(df)
        pred = model.predict(exampels)
        acc = sum(labels == pred) / len(labels)
        if acc > best_acc:
            best_acc=acc
            best_model=model
    return best_model
