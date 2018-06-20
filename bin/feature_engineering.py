#!/usr/bin/env python3

import numpy as np
import logging
import timeit
from librosa.feature import (zero_crossing_rate, mfcc, spectral_centroid,
                             spectral_rolloff, spectral_bandwidth, rmse
                             )


__all__ = [
    'FeatureEngineer'
]


class Features:
    """
    Feature engineering
    """

    #RATE = 44100   # All recordings are 44.1 kHz
    FRAME = 512    # Frame size in samples
    VECLENGTH = 220160

    def __init__(self, raw_audio_tuple):
        self.label = raw_audio_tuple[0]
        self.audio_data = raw_audio_tuple[1]
        self.rate = raw_audio_tuple[2]

    def engineer_features(self):
        """
        Extract features using librosa.feature.
        Each signal is cut into frames, features are computed for each frame and averaged [median].
        The numpy array is transformed into a data frame with named columns.
        :param audio_data: the input signal samples with frequency 44.1 kHz
        :return: a numpy array (numOfFeatures x numOfShortTermWindows)
        """

        logging.info('Computing Zero Crossing Rate...')
        start = timeit.default_timer()

        zcr_feat = zero_crossing_rate(y=self.audio_data, hop_length=self.FRAME)

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        logging.info('Computing RMSE ...')
        start = timeit.default_timer()

        rmse_feat = rmse(y=self.audio_data, hop_length=self.FRAME)

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        logging.info('Computing MFCC...')
        start = timeit.default_timer()

        mfcc_feat = mfcc(y=self.audio_data, sr=self.rate, n_mfcc=13)

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        logging.info('Computing spectral centroid...')
        start = timeit.default_timer()

        spectral_centroid_feat = spectral_centroid(y=self.audio_data,
                                                   sr=self.rate,
                                                   hop_length=self.FRAME)

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        logging.info('Computing spectral roll-off ...')
        start = timeit.default_timer()

        spectral_rolloff_feat = spectral_rolloff(y=self.audio_data,
                                                 sr=self.rate,
                                                 hop_length=self.FRAME,
                                                 roll_percent=0.90)

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        logging.info('Computing spectral bandwidth...')
        start = timeit.default_timer()

        spectral_bandwidth_feat = spectral_bandwidth(y=self.audio_data,
                                                     sr=self.rate,
                                                     hop_length=self.FRAME)

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        # print('zcr: ',zcr_feat.shape)
        # print('rmse: ',rmse_feat.shape)
        # print('mfcc: ',mfcc_feat.shape)
        # print('spect centroid: ', spectral_centroid_feat.shape)
        # print('spect rolloff: ', spectral_rolloff_feat.shape)
        # print('spectral_bandwidth_feat: ', spectral_bandwidth_feat.shape)

        concat_feat = np.concatenate((zcr_feat,
                                      rmse_feat,
                                      spectral_centroid_feat,
                                      spectral_rolloff_feat,
                                      spectral_bandwidth_feat,
                                      mfcc_feat,
                                      ), axis=0)

        logging.info('Averaging...')
        start = timeit.default_timer()

        mean_feat = np.mean(concat_feat, axis=1, keepdims=True).transpose()[0]

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        return mean_feat, self.label, concat_feat