#!/usr/bin/env python3
"""
coding=utf-8
Code Template
"""
import pickle
import logging
import os

import numpy as np
import librosa

import infer
import lib
import process_livestream_audio


def main():
    """

    :return: None
    :rtype: None
    """
    logging.basicConfig(level=logging.DEBUG)

    list_ = process_livestream_audio\
        .process_livestream_folder(audio_folder=AUDIO_FOLDER)

    pass


# Main section
if __name__ == '__main__':
    main()
