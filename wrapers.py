import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import KFold


class major_wraper:
    def __init__(self, model):
        self.model = model

    def fit_train(self, x_train_set: pd.DataFrame, y_train_set: pd.DataFrame):
        self.model.fit(x_train_set, y_train_set)

    def pred_label(self, pred_set: pd.DataFrame):
        probs_predictions = self.model.predict_proba(pred_set)
        results = np.mean(probs_predictions, axis=0)
        return np.argmax(results)


class probably_vote:
    def __init__(self, model, threshold: int):
        self.model = model
        self.threshold = threshold

    def fit_train(self, x_train_set: pd.DataFrame, y_train_set: pd.DataFrame):
        self.model.fit(x_train_set, y_train_set)

    def prob_vote(self, df: pd.DataFrame, party: int):
        will_vote = []
        probs_predictions = self.model.predict_proba(df)
        for idx, row in enumerate(probs_predictions):
            if row[party] > self.threshold:
                will_vote.append(idx)
        return df.iloc[will_vote, :]


def split_lab_exam(df: pd.DataFrame):
    featuers = []
    for col in df:
        if col != "Vote":
            featuers.append(col)
    labels = df["Vote"]
    exampels = df[featuers]
    return labels, exampels
