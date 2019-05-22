import numpy as np
import pandas as pd


class ElectionsResultsWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, x_train_set: pd.DataFrame, y_train_set: pd.DataFrame):
        self.model.fit(x_train_set, y_train_set)

    def predict(self, pred_set: pd.DataFrame):
        probs_predictions = self.model.predict_proba(pred_set)
        results = np.sum(probs_predictions, axis=0)
        return np.argmax(results)

    def predict_proba(self, pred_set: pd.DataFrame):
        probs_predictions = self.model.predict_proba(pred_set)
        return np.sum(probs_predictions, axis=0)


class LikelyVotersWrapper:
    def __init__(self, model, threshold: float):
        assert 0 < threshold < 1.0
        self.model = model
        self.threshold = threshold

    def fit(self, x_train_set: pd.DataFrame, y_train_set: pd.DataFrame):
        self.model.fit(x_train_set, y_train_set)

    def predict(self, df: pd.DataFrame, party: int):
        will_vote = []
        probs_predictions = self.model.predict_proba(df)
        for idx, row in enumerate(probs_predictions):
            if row[party] > self.threshold:
                will_vote.append(idx)
        return will_vote

