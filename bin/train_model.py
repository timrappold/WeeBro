import logging
import os
import importlib


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


import training_data
import features
import lib

importlib.reload(training_data)
importlib.reload(features)

#logging.basicConfig(level=logging.DEBUG)

MODEL_DIR = './trained_models/'


def get_scaler_svc():

    logging.debug("Extract: Get training data via training_data.py...")

    X_train, X_test, y_train, y_test = training_data.get_train_test_split(
        from_pickle=True,
        format='vec',
        test_size=0.20)

    logging.debug("Load: transform and fit data...")

    scaler = StandardScaler().fit(X_train)  # FIT
    X_train_scaled = scaler.transform(X_train)

    parameters = {'kernel': ('linear', 'rbf'), 'C': [0.01, 0.1, 1, 10, 100]}

    clf = GridSearchCV(SVC(), parameters)
    clf.fit(X_train_scaled, y_train)  # FIT

    logging.debug("Validating model...")
    X_test_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))

    trained_model = {'clf': clf,
                     'scaler': scaler,
                     'classification_report': classification_report}

    lib.dump_to_pickle(trained_model, MODEL_DIR + 'scaler_svc.pkl')

    return scaler, clf, classification_report


def get_pipe_svc():
    pass


def main():
    scaler, clf, classification_report = get_scaler_svc()
    return scaler, clf, classification_report


if __name__ == '__main__':

    scaler, clf, classification_report = main()

    logging.basicConfig(level=logging.DEBUG)

    test_folder = '../data/unit_test_data/'

    for file in os.listdir(test_folder):
        if file != '.DS_Store':
            f = features.Features(test_folder + file)
            f.featurize()

            if f.raw is not None:
                logging.debug(f'\nNew file: {file}')
                logging.debug(f'Raw: {f.raw.shape}')
                logging.debug(f'Vec: {f.vec.shape}')
                logging.debug(f'Mat: {f.mat.shape}')

                prediction = clf.predict(scaler.transform(f.vec))
                logging.info(f'Prediction: {prediction}')

            else:
                logging.debug(f"Did not featurize {file} because it is too short.")




