#!/usr/bin/env python3

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
    Converts a sound file (.wav, .ogg, etc) to a standard-length raw vector, a
    feature matrix, and a feature vector, which is the time average of the
    feature matrix.

    """

    RATE = 44100   # All recordings are 44.1 kHz
    FRAME = 512    # Frame size in samples
    VECLENGTH = 218111  # ~99% of the mode of 220160 for 5 sec samples and
                        # multiple of FRAME = 512.
    N_MFCC = 13  # number of Mel Frequency Cepstral Coefficients

    def __init__(self, path):
        """
        Cut all wav vectors to the same length to enforce uniformity.
        :param raw_audio_tuple:
        """
        self.path = path

        raw, _ = librosa.load(path, sr=self.RATE, mono=True)
        self.raw_audio_vec = raw[:self.VECLENGTH]

    def engineer_features(self):
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
        :param raw_audio_vec: the input signal samples with frequency 44.1 kHz
        :return: a numpy array (numOfFeatures x numOfShortTermWindows)
        """

        logging.info('Computing Zero Crossing Rate...')
        start = timeit.default_timer()

        zcr_feat = zero_crossing_rate(y=self.raw_audio_vec,
                                      hop_length=self.FRAME)

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        logging.info('Computing RMSE ...')
        start = timeit.default_timer()

        rmse_feat = rmse(y=self.raw_audio_vec, hop_length=self.FRAME)

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        logging.info('Computing MFCC...')
        start = timeit.default_timer()

        mfcc_feat = mfcc(y=self.raw_audio_vec,
                         sr=self.RATE,
                         n_mfcc=self.N_MFCC)

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        logging.info('Computing spectral centroid...')
        start = timeit.default_timer()

        spectral_centroid_feat = spectral_centroid(y=self.raw_audio_vec,
                                                   sr=self.RATE,
                                                   hop_length=self.FRAME)

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        logging.info('Computing spectral roll-off ...')
        start = timeit.default_timer()

        spectral_rolloff_feat = spectral_rolloff(y=self.raw_audio_vec,
                                                 sr=self.RATE,
                                                 hop_length=self.FRAME,
                                                 roll_percent=0.90)

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        logging.info('Computing spectral bandwidth...')
        start = timeit.default_timer()

        spectral_bandwidth_feat = spectral_bandwidth(y=self.raw_audio_vec,
                                                     sr=self.RATE,
                                                     hop_length=self.FRAME)

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        feature_matrix = np.concatenate((zcr_feat,
                                         rmse_feat,
                                         spectral_centroid_feat,
                                         spectral_rolloff_feat,
                                         spectral_bandwidth_feat,
                                         mfcc_feat,
                                         ), axis=0)

        logging.info('Averaging...')
        start = timeit.default_timer()

        feature_vector = np.mean(feature_matrix, axis=1, keepdims=True).transpose()[0]

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        return feature_vector, feature_matrix
