import os

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

import feature_engineering
import lib

# importlib.reload(feature_engineering)


def get_train_test_set(feature_version='feature_vector', from_pickle=True,
                       test_size=0.2):
    """
    Generate (from_pickle=False) or retrieve (from_pickle=True) training and
    test sets for model training and validation. Transforms raw sound file
    data into Feature Engineered matrices and vectors. Flavor of transform type
    is set with transform_type.

    :param feature_version: str. Either 'feature_vector' (preferred for linear
    model), 'feature_matrix', or 'raw_vector'.

    :param from_pickle: bool. If False, will generate new test_train sets and
    dump to pickle, overwriting current version.

    :param test_size: test_size for sklearn's train_test_split()

    :return: train_test_split (sklearn)
    """

    pickle_path = '../data/pickles/train_test_sets.pkl'

    if not from_pickle:
        sound_folder_root = '../data/trial_data/'
        subfolders = ['crying', 'silence', 'noise', 'baby_laugh',
                      'aria_crying', 'aria_other']
        folder_paths = [sound_folder_root + subfolder + '/' for subfolder in
                        subfolders]

        # cols for feature_vec_dicts:
        left_cols = ['zcr',
                     'rmse',
                     'spectral_centroid',
                     'spectral_rolloff',
                     'spectral_bandwidth',
                     ]

        cols_mfcc = ['mfcc_' + str(num)
                     if len(str(num)) == 2
                     else 'mfcc_0' + str(num)
                     for num in range(13)
                     # assume fixed num=13 of MFCCs
                     ]  # list of strings for DF header

        cols = ['label'] + left_cols + cols_mfcc

        feature_vec_dicts = []
        feature_mat_dicts = []
        raw_vec_dicts = []

        for folder in folder_paths:

            for file in os.listdir(folder):

                if file != '.DS_Store':

                    f = feature_engineering.Features(folder + file)
                    raw_vector = f.raw_audio_vec  # raw flt point time series

                    if len(raw_vector) == f.VECLENGTH:  # filters out truncated
                                                        #  sound files.

                        # feature vector is the 1D time average of feature
                        # matrix
                        feature_vector, feature_matrix = f.engineer_features()

                        label = int('crying' in folder)  # labeling by folder
                                                         # name

                        feature_vec_dicts.append(dict(zip(cols, np.concatenate(
                            ([label], feature_vector)))))

                        feature_mat_dicts.append((label, feature_matrix))

                        raw_vec_dicts.append(
                            np.concatenate(([label], raw_vector)))

        # create dictionary that contains the three flavors of data set, ready
        # for pickling:
        pickle_dict = dict()
        pickle_dict['feature_vector'] = feature_vec_dicts
        pickle_dict['feature_matrix'] = feature_mat_dicts
        pickle_dict['raw_vector'] = raw_vec_dicts

        lib.dump_pickle(pickle_dict, pickle_path)

    else:
        pickle_dict = lib.load_pickle(pickle_path)

    df = pd.DataFrame(pickle_dict[feature_version])

    # Assume that zeroth column is always the target:
    target_col = 0
    X = df.drop(df.columns[target_col], axis=1)
    y = df[df.columns[target_col]]

    return train_test_split(X, y, test_size=test_size)


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = get_train_test_set(
            feature_version='feature_vector', from_pickle=True)
