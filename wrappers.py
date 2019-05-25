import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class ElectionsResultsWrapper(BaseEstimator):
    def __init__(self, model):
        self.model = model
        self.targets = None

    def fit(self, x_train_set: pd.DataFrame, y_train_set: pd.DataFrame):
        self.model.fit(x_train_set, y_train_set)
        self.targets = y_train_set.unique()
        self.targets.sort()

    def predict(self, pred_set: pd.DataFrame):
        return self.predict_proba(pred_set).idxmax()

    def predict_proba(self, pred_set: pd.DataFrame):
        probs_predictions = self.model.predict_proba(pred_set)
        return pd.Series(np.sum(probs_predictions, axis=0), index=self.targets)


class LikelyVotersWrapper(BaseEstimator):
    def __init__(self, model, threshold: float = 0.75):
        assert 0 < threshold < 1.0
        self.model = model
        self.threshold = threshold
        self.targets = None

    def fit(self, x_train_set: pd.DataFrame, y_train_set: pd.Series):
        self.model.fit(x_train_set, y_train_set)
        self.targets = y_train_set.unique()
        self.targets.sort()

    def predict(self, df: pd.DataFrame, party: str):
        probs_predictions = self.model.predict_proba(df)
        party_idx = np.where(self.targets == party)[0].item()

        return df.index[np.where(probs_predictions[:, party_idx] > self.threshold)]
