import numpy as np


class Dataset:
    def __init__(self , matrix , features):
        self.matrix = matrix
        self.features = features

    def __repr__(self):
        return str(self.features) + '\n' + str(self.matrix)


def testTrainSplit(x, testsize):
    size = len(x)
    nTestSamples = int(size * testsize)
    nTrainSamples = size - nTestSamples
    np.random.shuffle(x)
    testSet = x[:nTestSamples]
    trainSet = x[nTestSamples:]
    return trainSet, testSet