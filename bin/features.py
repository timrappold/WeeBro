#!/usr/bin/env python3

"""
Author: Tim Rappold

This script contains the class Features, which processes all raw sound files
(*.wav, *.ogg, likely other formats as well) into 1D and 2D tensors that can
be used by machine learning applications.

"""

import os
import numpy as np
import logging
import timeit
import librosa
from librosa.feature import (zero_crossing_rate, mfcc, spectral_centroid,
                             spectral_rolloff, spectral_bandwidth, rmse
                             )


class Features(object):
    """
    Feature engineering for linear machine learning models and neural networks.
    Converts a sound file (.wav, .ogg, etc) to a standard-length raw vector
    (raw), a feature matrix (mat), and a feature vector (vec), which is the
    time average of the feature matrix.

    """

    RATE = 44100   # All recordings are 44.1 kHz
    FRAME = 512    # Frame size in samples
    TRUNCLENGTH = 218111  # ~99% of the mode of 220160 for 5 sec samples and
                        # multiple of FRAME = 512.
    N_MFCC = 13  # number of Mel Frequency Cepstral Coefficients

    def __init__(self, path):
        """
        :param path: str. Path to sound file (*.wav, *.ogg).
        """
        self.path = path
        self.raw = None
        self.vec = None
        self.mat = None

    def featurize(self):
        """
        Extract features using librosa.feature. Convert wav vec, the sound
        amplitude as a function of time, to a variety of extracted features,
        such as Mel Frequency Cepstral Coeffs, Root Mean Square Energy, Zero
        Crossing Rate, etc.

        :param observations
        :ptype: list of tuples (label, wav vec, sampling rate)
        :return:
        :rtype:

        Each signal is cut into frames, features are computed for each frame and averaged [median].
        The numpy array is transformed into a data frame with named columns.
        :param raw: the input signal samples with frequency 44.1 kHz
        :return: a numpy array (numOfFeatures x numOfShortTermWindows)
        """

        start = timeit.default_timer()

        logging.debug('Loading Librosa raw audio vector...')

        raw, _ = librosa.load(self.path, sr=self.RATE, mono=True)
        raw = raw[:self.TRUNCLENGTH]

        if len(raw) < self.TRUNCLENGTH:
            logging.info(f"Not featurizing {self.path} because raw vector is "
                         f"too short. `None` will be returned for all data "
                         f"formats.")
            return self

        logging.debug('Computing Zero Crossing Rate...')
        zcr_feat = zero_crossing_rate(y=raw,
                                      hop_length=self.FRAME)

        logging.debug('Computing RMSE ...')
        rmse_feat = rmse(y=raw, hop_length=self.FRAME)

        logging.debug('Computing MFCC...')
        mfcc_feat = mfcc(y=raw,
                         sr=self.RATE,
                         n_mfcc=self.N_MFCC)


        logging.debug('Computing spectral centroid...')
        spectral_centroid_feat = spectral_centroid(y=raw,
                                                   sr=self.RATE,
                                                   hop_length=self.FRAME)


        logging.debug('Computing spectral roll-off ...')
        spectral_rolloff_feat = spectral_rolloff(y=raw,
                                                 sr=self.RATE,
                                                 hop_length=self.FRAME,
                                                 roll_percent=0.90)


        logging.debug('Computing spectral bandwidth...')
        spectral_bandwidth_feat = spectral_bandwidth(y=raw,
                                                     sr=self.RATE,
                                                     hop_length=self.FRAME)

        logging.debug('Concatenate all features...')
        mat = np.concatenate((zcr_feat,
                              rmse_feat,
                              spectral_centroid_feat,
                              spectral_rolloff_feat,
                              spectral_bandwidth_feat,
                              mfcc_feat,
                              ), axis=0)

        logging.debug(f'Mat shape: {mat.shape}')

        logging.debug(f'Create self.raw...')
        self.raw = raw.reshape(1, -1)

        logging.debug(f'Create self.vec by averaging mat along time dim...')
        self.vec = np.mean(mat, axis=1, keepdims=True).reshape(1, -1)

        logging.debug(f'Vec shape: {self.vec.shape}')

        logging.debug(f'Create self.mat...')
        assert mat.shape == (18, 426), 'Matrix dims do not match (426,18)'
        self.mat = mat.reshape(1, 18, 426,)

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))


        return self

if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    test_folder = '../data/unit_test_data/'

    for file in os.listdir(test_folder):
        if file != '.DS_Store':
            f = Features(test_folder + file)
            f.featurize()
            if f.raw is not None:
                logging.debug(f'\nNew file: {file}')
                logging.debug(f'Raw: {f.raw.shape}')
                logging.debug(f'Vec: {f.vec.shape}')
                logging.debug(f'Mat: {f.mat.shape}')
            else:
                logging.debug(f"Did not featurize {file} because it is too short.")

            #prediction_results.append((file, predict2(folder + file, grid)))




