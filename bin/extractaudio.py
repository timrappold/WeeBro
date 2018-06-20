# -*- coding: utf-8 -*-

#from feature_extraction import features
from os import listdir
import librosa
import pickle


# = [sound_folder_path + str(f) for f in listdir(nb_path)]



class AudioLoader(object):
    """
    Read input audio file for training set
    file_name: 'path/to/file/filename.ogg OR .wav'
    """

    #SAMPLING_RATE = 44100

    def __init__(self, file_name):
        self.file_name = file_name
        pass

    def read_audio_file(self):
        """
        Read audio file using librosa.load.

        :return: audio_data (numpy.ndarray). 1-D array of the raw audio signal
            in time.
        :rtype: numpy.ndarray

        :return: sr (sampling rate in Hz)
        :rtype: int
        """




        audio_data, sr = librosa.load(self.file_name,
                                      sr=44100,
                                      mono=True,
                                      duration=5)

        return audio_data, sr



class RawAudioExtractor(object):
    """
    Extract raw audio data from .wav/.ogg files or from pickle.
    """

    ROOTPATH =

    def __init__(self, from_pickle=True):
        self.from_pickle_ = from_pickle
        pass

    def load_from_audio_files(self):
        raw_audio_data = []
        for folder in folder_paths:

            for file in os.listdir(folder):
                if file != '.DS_Store':
                    raw_audio_data.append((int('crying' in folder),
                                           librosa.load(folder + file,
                                                        sr=44100)))


        self.raw_audio_data = raw_audio_data

        with open(raw_audio_data.pkl, 'wb') as picklefile:
            pickle.dump(data, picklefile)
        return

    def load_from_pickle(self):




def main():

    return None


if __name__ == '__main__':
    main()
