import numpy as np
from preprocessors.discritizer import binarizer
from algorithms.decision_tree.decision_tree import DecisionTree


def creditDataset():
    FILE = 'D:/Education/L4T2/0dipto_L4T2/ML Sessional/AdaBoost/datasets/creditcard.csv'
    fp = open(FILE)
    lines = fp.readlines()
    dataSet = []

    fraud = []
    for line in lines[1:]:
        line = line.strip().split(',')
        line = line[:-1] + [line[-1][1]]
        line = list(map(float,line))
        if line[-1] == 1: fraud.append(line)
        else : dataSet.append(line)
    dataSet = np.array(dataSet)
    np.random.shuffle(dataSet)

    for x in dataSet:
    # for x in dataSet[:int(len(lines) * 10)]:
        fraud.append(x)

    dataSet = np.array(fraud[:30000])

    # encoding class labels
    labels = list(set(dataSet[:, -1]))
    encodeLabel = {}
    i = 0
    for l in labels:
        encodeLabel[l] = i
        i += 1
    labels = [encodeLabel[x] for x in dataSet[:, -1]]

    # processing continous attributes
    for i in range(len(dataSet[0])-1):
        contData = list(map(float, dataSet[:, i]))
        contDataTuples = []
        for j in range(len(labels)):
            contDataTuples.append((contData[j], labels[j]))
        splitPoint = binarizer(contDataTuples)
        for j in range(len(dataSet)):
            if float(dataSet[j][i]) <= splitPoint:
                dataSet[j][i] = 0
            else:
                dataSet[j][i] = 1

    # return 28000 samples -> 1/10 th of total data
    return dataSet[:28000]







def getStats(trueLabels, predictedLabels):
    truePositive = sum([trueLabels[i] == predictedLabels[i] == 1 for i in range(len(trueLabels))])
    falseNegatives = sum([trueLabels[i] == 1 and predictedLabels[i] == 0 for i in range(len(trueLabels))])
    trueNegative = sum([trueLabels[i] == predictedLabels[i] == 0 for i in range(len(trueLabels))])
    falsePositives = sum([trueLabels[i] == 0 and predictedLabels[i] == 1 for i in range(len(trueLabels))])
    # truePositiveRate or sensitivity or TPR
    TPR = truePositive/(truePositive+falseNegatives)
    # trueNegativeRate or specificity or TNR
    TNR = trueNegative / (trueNegative + falsePositives)
    # precision or positive predictive value
    PPV = truePositive/(truePositive+falsePositives)
    # false discovery rate (FDR)
    FDR = 1-PPV
    # F1 score
    F1 = (2*truePositive)/((2*truePositive)+falsePositives+falseNegatives)
    print('truePositiveRate or sensitivity or TPR = ',TPR)
    print('trueNegativeRate or specificity or TNR = ',TNR)
    print('precision or positive predictive value = ',PPV)
    print('false discovery rate (FDR) = ', FDR)
    print('F1 score = ', F1)















if __name__ == '__main__':

    dataSet = creditDataset()
    np.random.shuffle(dataSet)
    trainSize = int(len(dataSet)*0.8)
    testSize = len(dataSet) - trainSize


    trainSet = dataSet[:trainSize]
    testSet = dataSet[trainSize: ]



    model = DecisionTree()
    model.train(trainSet)

    print('Dataset : Creditcard Fraud')
    print('---------------------------\n')






    print('TrainSet Statistics')
    print('-------------------')
    # on TrainSet
    error = 0
    predictedLabels = []
    for s in trainSet:
        pred = model.predict(s)
        predictedLabels.append(pred)
        if pred != s[-1]: error += 1
    acc = (1 - (error/len(trainSet)))
    print('Trainset Accuracy = ' , acc)
    getStats(trainSet[:,-1], predictedLabels)


    print('\n\n\n\nTestSet Statistics')
    print('-------------------')
    # on TestSet
    error = 0
    predictedLabels = []
    for s in testSet:
        pred = model.predict(s)
        predictedLabels.append(pred)
        if pred != s[-1]: error += 1
    acc = (1 - (error/len(testSet)))
    print('Testset Accuracy = ' , acc)
    getStats(testSet[:,-1], predictedLabels)

