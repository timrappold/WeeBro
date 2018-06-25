from datetime import datetime
import os

import infer
import lib


AUDIO_FOLDER = '../data/live_stream_conversions/'


def filename_to_datetime(filename):
    """
    Convert file name of the format 'audio_chunk_YYYY-MM-DD_HH-MM-SS.wav' to
    datetime object.
    :param filename: str.
    :return: datetime.
    """

    return datetime.strptime(filename[12:-4].replace('-', ''), '%Y%m%d_%H%M%S')


def process_livestream_file(path, infer):
    filename = os.path.basename(path)

    timestamp = filename_to_datetime(filename)

    prediction = infer(path)

    return tuple((timestamp, prediction))


def process_livestream_folder(audio_folder=AUDIO_FOLDER):

    print(f'Processing {audio_folder}...')
    infer_from_wav = infer.make_infer()  # Create inference function

    pkl_name = 'total_stream.pkl'

    list_ = lib.load_pickle(
        '../data/live_stream_predictions/'+pkl_name)

    for file in os.listdir(audio_folder):
        if file.endswith('.wav'):
            print(f'File: {file}')

            pred_ = process_livestream_file(audio_folder+file, infer_from_wav)
            print(pred_)

            list_.append(pred_)

            # Move processed file to archive:
            os.rename(audio_folder+file, audio_folder+'archive/'+file)

    lib.dump_to_pickle(list_, '../data/live_stream_predictions/'+pkl_name)

    return list_


def main():

    process_livestream_folder(audio_folder=AUDIO_FOLDER)
    pass


if __name__ == '__main__':

    main()
