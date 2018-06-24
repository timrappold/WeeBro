#!/usr/bin/env python3

"""
Author: Tim Rappold

This script contains functions that prepare training and model validation
data.

The dictionary training_data can be obtained from the original training
data => prepare_training_data() or from a pickle file => get_training_data_dict()

get_train_test_split() will return X_train, X_test, y_train, y_test using either
of the above two functions.

"""

import os
import logging

import numpy as np
from sklearn.model_selection import train_test_split

import features
import lib

import importlib


PICKLE_PATH = '../data/pickles/train_test_sets.pkl'


def prepare_training_data():
    """
    Create a training and test sets.

    :param from_pickle:
    :return: sklearn.model_selection.train_test_split()
    """

    sound_root = '../data/training_data/'

    sound_folders = ['crying', 'livestream_crying', 'silence', 'noise',
                     'baby_laugh', 'aria_crying', 'aria_other']

    label_list = []
    raw_list = []
    vec_list = []
    mat_list = []

    for folder in sound_folders:

        logging.info(f'Processing files in {sound_root + folder}...')

        label = int('crying' in folder)  # data labels are determined by
                                         # folder name.

        path = os.path.join(sound_root, folder)

        for file in os.listdir(path):

            logging.debug(f'Processing {file} in folder {folder}...')

            if file.endswith(('.wav', '.ogg')):

                file_path = os.path.join(path, file)

                f = features.Features(file_path).featurize()

                if f.raw is None:
                    print(f'Skipping {file} because it is likely too short')
                    continue

                label_list.append(np.array([label]))
                raw_list.append(f.raw)
                vec_list.append(f.vec)
                mat_list.append(f.mat)

    # TODO: Re-Write interface for mat_list elements to fit with 1D convnet.

    training_data_ = dict()
    training_data_['label'] = np.concatenate(label_list)
    training_data_['raw'] = np.concatenate(raw_list)
    training_data_['vec'] = np.concatenate(vec_list)
    training_data_['mat'] = np.concatenate(mat_list)

    lib.dump_to_pickle(training_data_, PICKLE_PATH)

    return training_data_


def get_training_data_dict():
    """
    Load training_data from pickle.
    :return: dict. keys: 'label', 'raw', 'vec', 'mat'
    """
    return lib.load_pickle(PICKLE_PATH)


def get_train_test_split(from_pickle=False, format='vec', test_size=0.25):
    """
    Get the sklearn train_test_split by processing the training-data dictionary.
    Choose one of three formats:
        'raw' (num samples, 218111): raw, time-resolved sound amplitude vector.
        'vec' (num samples, 18) : a composite of 18 librosa features, average over time
        'mat' (num samples, 18, 426): same as above, but not averaged.

    :param from_pickle: bool.
    :param format: str. 'raw', 'vec', or 'mat'. 'vec' is default.
    :param test_size: float. Same as test_size in sklearn.(...).train_test_split.
    :return:
    """

    if from_pickle:
        training_data_ = get_training_data_dict()
    else:
        training_data_ = prepare_training_data()

    return train_test_split(training_data_[format],
                            training_data_['label'],
                            test_size=test_size)


if __name__ == '__main__':

    importlib.reload(lib)
    importlib.reload(features)

    logging.basicConfig(level=logging.DEBUG)
    #

    X_train, X_test, y_train, y_test = get_train_test_split(from_pickle=False,
                                                            format='vec',
                                                            test_size=0.25)

    training_data = get_training_data_dict()

    for key, value in training_data.items():
        print(key, value.shape)

    pass
