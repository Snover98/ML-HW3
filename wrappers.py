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
        probs_predictions = self.model.predict_proba(df)

        return [idx for count, (idx, _) in enumerate(df.iterrows()) if probs_predictions[count] > self.threshold]
