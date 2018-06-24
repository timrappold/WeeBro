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


MODEL_DIR = './trained_models/'


def get_trained_model(file=MODEL_DIR+'scaler_svc.pkl'):
    trained_model = lib.load_pickle(file)
    scaler = trained_model['scaler']
    clf = trained_model['clf']
    return scaler, clf


def infer(vec, clf, scaler=None):

    print("Infer vec shape: ", vec.shape)
    if scaler is not None:
        return clf.predict(scaler.transform(vec))
    else:
        return clf.predict(vec)


def infer_test_files(clf, scaler=None):
    test_folder = '../data/unit_test_data/'

    for file in os.listdir(test_folder):
        if file != '.DS_Store':
            f = features.Features(test_folder + file)
            f.featurize()

            if f.raw is not None:
                print(f'\nNew file: {file}')
                logging.debug(f'Raw: {f.raw.shape}')
                #print(f'Vec: {f.vec.shape}')
                logging.debug(f'Mat: {f.mat.shape}')

                prediction = infer(f.vec, clf, scaler=scaler)

                print(f'Prediction: {prediction}')

            else:
                logging.debug(f"Did not featurize {file} because it is too short.")


def infer_training_data(clf, scaler=None):

    training_data_ = training_data.get_training_data_dict()

    for sample in training_data_['vec']:

        #print(sample.shape)
        print(infer(sample.reshape(1, -1), clf, scaler=scaler))

    pass


def infer_livestream(clf, scaler=None):
    test_folder = '../data/live_stream_conversions/'

    livestream_predictions = []
    for file in os.listdir(test_folder):
        if file != '.DS_Store':
            f = features.Features(test_folder + file)
            f.featurize()

            if f.raw is not None:
                print(f'\nNew file: {file}')
                logging.debug(f'Raw: {f.raw.shape}')
                #print(f'Vec: {f.vec.shape}')
                logging.debug(f'Mat: {f.mat.shape}')

                prediction = infer(f.vec, clf, scaler=scaler)

                print(f'Prediction: {prediction}')

                livestream_predictions.append(prediction)

            else:
                logging.debug(f"Did not featurize {file} because it is too short.")

    return livestream_predictions


def main():

    scaler, clf = get_trained_model(file=MODEL_DIR + 'scaler_svc.pkl')

    infer_training_data(clf, scaler=scaler)

    pass


if __name__ == '__main__':

    main()

    #print('Training data:')
    #infer_training_data(clf, scaler=scaler)

    #print('Test files:')
    #infer_test_files(clf, scaler=scaler)

    #print('Livestream data:')

    #scaler, clf, classification_report = train_model.main()

    #predictions = infer_livestream(clf, scaler=scaler)



