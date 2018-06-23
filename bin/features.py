#!/usr/bin/env python3

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
        Cut all wav vectors to the same length to enforce uniformity.
        :param raw_audio_tuple:
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

        raw, _ = librosa.load(self.path, sr=self.RATE, mono=True)
        raw = raw[:self.TRUNCLENGTH]

        if len(raw) < self.TRUNCLENGTH:
            logging.info(f"Not featurizing {self.path} because vector is too short.")
            return self

        logging.info('Computing Zero Crossing Rate...')
        start = timeit.default_timer()

        zcr_feat = zero_crossing_rate(y=raw,
                                      hop_length=self.FRAME)

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        logging.info('Computing RMSE ...')
        start = timeit.default_timer()

        rmse_feat = rmse(y=raw, hop_length=self.FRAME)

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        logging.info('Computing MFCC...')
        start = timeit.default_timer()

        mfcc_feat = mfcc(y=raw,
                         sr=self.RATE,
                         n_mfcc=self.N_MFCC)

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        logging.info('Computing spectral centroid...')
        start = timeit.default_timer()

        spectral_centroid_feat = spectral_centroid(y=raw,
                                                   sr=self.RATE,
                                                   hop_length=self.FRAME)

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        logging.info('Computing spectral roll-off ...')
        start = timeit.default_timer()

        spectral_rolloff_feat = spectral_rolloff(y=raw,
                                                 sr=self.RATE,
                                                 hop_length=self.FRAME,
                                                 roll_percent=0.90)

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        logging.info('Computing spectral bandwidth...')
        start = timeit.default_timer()

        spectral_bandwidth_feat = spectral_bandwidth(y=raw,
                                                     sr=self.RATE,
                                                     hop_length=self.FRAME)

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        mat = np.concatenate((zcr_feat,
                              rmse_feat,
                              spectral_centroid_feat,
                              spectral_rolloff_feat,
                              spectral_bandwidth_feat,
                              mfcc_feat,
                              ), axis=0)

        logging.debug(f'Mat shape: {mat.shape}')

        self.raw = raw.reshape(1, -1)

        self.vec = np.mean(mat, axis=1, keepdims=True).reshape(1,-1)

        logging.debug(f'Vec shape: {self.vec.shape}')

        assert mat.shape == (18, 426), 'Matrix dims do not match (426,18)'
        self.mat = mat.reshape(1, 18, 426,)

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



