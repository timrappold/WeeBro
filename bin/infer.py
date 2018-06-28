import logging
import os
import importlib

import train_model
import training_data
import features
import lib


importlib.reload(train_model)
importlib.reload(training_data)
importlib.reload(features)


MODEL_DIR = './trained_models/' + 'scaler_svc.pkl'


def main():

    infer = make_infer()

    infer_test_files(infer)

    return None


def load_model():
    """
    Load dictionary that contains a trained cry-no cry classification model.
    Keys: 'clf', 'scaler', 'classification_report'.

    :return: dict.
    """

    return lib.load_pickle(MODEL_DIR)


def make_infer():
    """
    Make the function `prediction = infer(wav)`. Make_infer binds a trained
    model to the parameters in `infer`.

    :return: function.
    """

    trained_model = load_model()
    scaler = trained_model['scaler']
    clf = trained_model['clf']

    def infer(wav):

        f = features.Features(wav)
        f.featurize()

        if f.vec is None:
            print(f'\nFile {wav} is not featurized. File likely too short.')
            return None
        else:
            return clf.predict(scaler.transform(f.vec))

    return infer


def infer_test_files(infer):
    test_folder = '../data/unit_test_data/'

    for file in os.listdir(test_folder):
        if file != '.DS_Store':
            prediction = infer(test_folder + file)
            print(file, ': ', prediction)


def infer_training_data(clf, scaler=None):
    """
    A simple test function that only outputs print statements.
    :param clf:
    :param scaler:
    :return: None.
    """

    training_data_ = training_data.get_training_data_dict()

    for sample in training_data_['vec']:
        print(infer(sample.reshape(1, -1), clf, scaler=scaler))

    pass


def infer_livestream(infer):
    """

    :param infer:
    :return:
    """
    # test_folder = '../data/live_stream_conversions/'
    test_folder = '../data/dummy_live_stream_conversions/'

    livestream_predictions = []
    for file in os.listdir(test_folder):
        if file != '.DS_Store':
            prediction = infer(test_folder + file)
            livestream_predictions.append(prediction)

    return livestream_predictions


# def infer_livestream(clf, scaler=None):
#     test_folder = '../data/live_stream_conversions/'
#     test_folder = '../data/dummy_live_stream_conversions/'
#
#
#     livestream_predictions = []
#     for file in os.listdir(test_folder):
#         if file != '.DS_Store':
#             f = features.Features(test_folder + file)
#             f.featurize()
#
#             if f.raw is not None:
#                 print(f'\nNew file: {file}')
#                 logging.debug(f'Raw: {f.raw.shape}')
#                 #print(f'Vec: {f.vec.shape}')
#                 logging.debug(f'Mat: {f.mat.shape}')
#
#                 prediction = infer(f.vec, clf, scaler=scaler)
#
#                 print(f'Prediction: {prediction}')
#
#                 livestream_predictions.append(prediction)
#
#             else:
#                 logging.debug(f"Did not featurize {file} because it is too short.")
#
#     return livestream_predictions



if __name__ == '__main__':

    main()

    #print('Training data:')
    #infer_training_data(clf, scaler=scaler)

    #print('Test files:')
    #infer_test_files(clf, scaler=scaler)

    #print('Livestream data:')

    #scaler, clf, classification_report = train_model.main()

    #predictions = infer_livestream(clf, scaler=scaler)



