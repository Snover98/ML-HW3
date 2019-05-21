import numpy as np
import pandas as pd
import sklearn as sk


class wraper:
    def __init__(self, model):
        self.model = model

    def fit_train(self, x_train_set: pd.DataFrame, y_train_set: pd.DataFrame):
        self.model.fit(x_train_set, y_train_set)

    def pred_label(self, pred_set: pd.DataFrame):
        prediction = self.model.pred(pred_set)
        probs_predictions = self.model.predict_proba(pred_set)
        results = np.mean(probs_predictions, axis=0)
        return np.argmax(results)
