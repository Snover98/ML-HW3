import loading
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from imputation import *
import cleansing
from standartisation import *
from sklearn.tree import DecisionTreeClassifier
from pprint import pprint
import wrapers

def rows_in_df(df: pd.DataFrame, rows: List[int]) -> List[int]:
    return list(set(df.index.values.tolist()).intersection(set(rows)))


def main():
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
    train.to_csv("train_after_imputation.csv", index=False)

    imputation(valid, train)
    valid.to_csv("valid_after_imputation.csv", index=False)

    imputation(test, train)
    test.to_csv("test_after_imputation.csv", index=False)





if __name__ == "__main__":
    main()
