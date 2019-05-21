import loading
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from imputation import *
import cleansing
from feat_selection import *
from standartisation import *
from sklearn.tree import DecisionTreeClassifier
from pprint import pprint
from relief_algo import relief


def show_info(df: pd.DataFrame, stage: str):
    plt.imshow(df.corr().abs())
    plt.colorbar()
    plt.title(f'correlation after {stage}')
    plt.show()
    plt.close('all')

    plt.imshow(calc_MI_matrix(df))
    plt.colorbar()
    plt.title(f'MI after {stage}')
    plt.show()
    plt.close('all')


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

    # train = pd.read_csv('train_after_imputation.csv')
    # valid = pd.read_csv('valid_after_imputation.csv')

    show_info(train, "imputation")

    features: List[str] = train.columns.to_numpy().tolist()
    non_norm_feats = ["Looking_at_poles_results", "Married", "Will_vote_only_large_party", "Financial_agenda_matters"]
    non_norm_feats += [f for f in features if f.startswith('Occ') or f.startswith('trans') or f.startswith("Issue")]

    scaler = DFScaler(train, list(set(features).difference(set(non_norm_feats + ['Vote']))))

    features_close = remove_similar_features(train, 'Vote', 8)
    print(f'features after close removal:')
    pprint(features_close)
    show_info(train[features_close], "close removal")

    # features_far = remove_far_features(train, 'Vote', 8)
    # print(f'features after far removal:')
    # pprint(features_far)
    # show_info(train[features_far], "far removal")

    features_close_far = remove_similar_features(train[features_close], 'Vote', 8)
    print(f'features after close and then far removal:')
    pprint(features_close_far)
    show_info(train[features_close_far], "close and then far removal")

    # features_far_close = remove_far_features(train[features_far], 'Vote', 8)
    # print(f'features after far and then close removal:')
    # pprint(features_far_close)
    # show_info(train[features_far_close], "far and then close removal")

    train = scaler.scale(train)
    valid = scaler.scale(valid)
    test = scaler.scale(test)
    show_info(train, "scaling")

    relief_features = relief(train[features_close_far], 'Vote', 8)
    relief_features += ['Vote']
    print(f'features after relief:')
    pprint(relief_features)
    show_info(train[relief_features], "relief")

    sfs_features = wrapper_SFS(train[relief_features], valid[relief_features], 'Vote', DecisionTreeClassifier())
    sfs_features += ['Vote']
    print(f'features after SFS:')
    pprint(sfs_features)
    show_info(train[sfs_features], "SFS")

    sbs_features = wrapper_SBS(train[relief_features], valid[relief_features], 'Vote', DecisionTreeClassifier())
    sbs_features += ['Vote']
    print(f'features after SBS:')
    pprint(sbs_features)
    show_info(train[sbs_features], "SBS")

    # TODO choose final features
    final_features = sbs_features
    print("the final features are:")
    pprint(final_features)
    train[final_features].to_csv('train_final.csv', index=False)
    valid[final_features].to_csv('valid_final.csv', index=False)
    test[final_features].to_csv('test_final.csv', index=False)


if __name__ == "__main__":
    main()
