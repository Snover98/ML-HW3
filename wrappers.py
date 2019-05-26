import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class ElectionsResultsWrapper(BaseEstimator):
    def __init__(self, model):
        super(ElectionsResultsWrapper, self).__init__()
        self.model = model
        self.targets = None

    def fit(self, x_train_set: pd.DataFrame, y_train_set: pd.DataFrame):
        self.model.fit(x_train_set, y_train_set)
        self.targets = y_train_set.unique()
        self.targets.sort()

    def predict(self, pred_set: pd.DataFrame):
        probs_predictions = self.model.predict_proba(pred_set)
        return pd.Series(np.sum(probs_predictions, axis=0), index=self.targets)


class ElectionsWinnerWrapper(ElectionsResultsWrapper):
    def predict(self, pred_set: pd.DataFrame):
        return self.predict_res(pred_set).idxmax()

    def predict_res(self, pred_set: pd.DataFrame):
        return super(ElectionsWinnerWrapper, self).predict(pred_set)


class LikelyVotersWrapper(BaseEstimator):
    def __init__(self, model, threshold: float = 0.6):
        assert 0 < threshold < 1.0
        super(LikelyVotersWrapper, self).__init__()
        self.model = model
        self.threshold = threshold
        self.targets = None

    def fit(self, x_train_set: pd.DataFrame, y_train_set: pd.Series):
        self.model.fit(x_train_set, y_train_set)
        self.targets = y_train_set.unique()
        self.targets.sort()

    def _get_party_likely_voters(self, df: pd.DataFrame, probs_predictions: pd.DataFrame, party: str):
        return list(df.index[probs_predictions[party] > self.threshold])

    def predict(self, df: pd.DataFrame, party: str = None):
        probs_predictions = pd.DataFrame(self.model.predict_proba(df), columns=self.targets)

        if party is not None:
            return self._get_party_likely_voters(df, probs_predictions, party)

        likely_voters = {tar: self._get_party_likely_voters(df, probs_predictions, tar) for tar in self.targets}
        likely_voters.update({None: list(df.index.difference(sum(likely_voters.values(), [])))})

        return likely_voters
