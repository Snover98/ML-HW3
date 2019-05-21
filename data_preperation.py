import loading
import numpy as np
import pandas as pd
import sklearn
from imputation import *
import cleansing
from standartisation import *
from sklearn.tree import DecisionTreeClassifier


def prepare_data():
    df = loading.load_csv()

    # splitting the data
    train, valid, test = loading.split_data(df, 'Vote')

    train.to_csv('orig_train.csv', index=False)
    valid.to_csv('orig_valid.csv', index=False)
    test.to_csv('orig_test.csv', index=False)

    # cleansing the data
    train = pd.DataFrame(cleansing.cleanse(train))
    valid = pd.DataFrame(cleansing.cleanse(valid))
    test = pd.DataFrame(cleansing.cleanse(test))

    # imputation of the data
    imputation(train)
    imputation(valid, train)
    imputation(test, train)

    features: List[str] = train.columns.to_numpy().tolist()
    selected_features = ["Avg_environmental_importance", "Avg_government_satisfaction", "Avg_education_importance",
                         "Avg_monthly_expense_on_pets_or_plants", "Avg residancy altitude", "Yearly_ExpensesK",
                         "Weighted_education_rank", "Number_of_valued_Kneset_members"]
    features = [feat for feat in features if feat.startswith("Issue")] + selected_features

    scaler = DFScaler(train, features)

    train = scaler.scale(train)
    valid = scaler.scale(valid)
    test = scaler.scale(test)

    train[features].to_csv('train_processed.csv', index=False)
    valid[features].to_csv('valid_processed.csv', index=False)
    test[features].to_csv('test_processed.csv', index=False)

    return train[features], valid[features], test[features]


if __name__ == "__main__":
    prepare_data()
