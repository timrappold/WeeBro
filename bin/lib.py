import pickle


def dump_to_pickle(data, filename):
    with open(filename, 'wb') as picklefile:
        pickle.dump(data, picklefile)


def load_pickle(filename):
    with open(filename, 'rb') as picklefile:
        return pickle.load(picklefile)
