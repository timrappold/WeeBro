#!/usr/bin/env python3

"""
Author: Tim Rappold
"""
import os
import logging

import numpy as np
from sklearn.model_selection import train_test_split

import features
import lib

import importlib


PICKLE_PATH = '../data/pickles/train_test_sets.pkl'


def label_raw_and_vec(label, unshaped_array):
    """
    Concatenate data label with 1D vector in get_train_test. Works only for
    label + raw and label + vec to convert dims (), (x,) -> (1, x+1). Does not
    work for mat.

    :param label: int.
    :param unshaped_array: np.array, dims (x,).
    :return:
    """
    label_ = np.array([[label]])
    vec_ = unshaped_array.reshape(1, -1)

    return np.concatenate((label_, vec_), axis=1)


def prepare_training_data():
    """
    Create a training and test sets.

    :param from_pickle:
    :return: sklearn.model_selection.train_test_split()
    """

    sound_root = '../data/training_data/'

    sound_folders = ['crying', 'silence', 'noise', 'baby_laugh',
                     'aria_crying', 'aria_other']

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


def get_train_test_dict():
    """
    Load training_data from pickle.
    :return: dict. keys: 'label', 'raw', 'vec', 'mat'
    """
    return lib.load_pickle(PICKLE_PATH)


def get_train_test(from_pickle=False, format='vec', test_size=0.25):
    """

    :param from_pickle: bool. If True
    :param format:
    :param train_test_split:
    :return:
    """

    if from_pickle:
        training_data_ = get_train_test_dict()
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

    X_train, X_test, y_train, y_test = get_train_test(from_pickle=False,
                                                      format='vec',
                                                      test_size=0.25)

    training_data = get_train_test_dict()

    for key, value in training_data.items():
        print(key, value.shape)

    pass
