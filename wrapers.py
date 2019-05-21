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


class pick_model:
    def __init__(self, models):
        self.models = models

    def fit_train(self, x_train_set: pd.DataFrame, y_train_set: pd.DataFrame):
        for model in self.models:
            model.fit(x_train_set, y_train_set)

    def choose_best(self, validation_set: pd.DataFrame,num_folds):
        best_acc = 0
        best_model = self.models[0]
        kf = KFold(n_splits=num_folds)
        for model in self.models:
            acc =
