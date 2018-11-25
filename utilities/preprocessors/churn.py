import numpy as np
from preprocessors.discritizer import binarizer
from algorithms.decision_tree.decision_tree import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier



def churnDataset():
    # we have to change this two only
    FILE = 'D:/Education/L4T2/0dipto_L4T2/ML Sessional/AdaBoost/datasets/telco.csv'
    continuousAttributeIndices = [4,17,18]

    fp = open(FILE)
    lines = fp.readlines()
    dataSet = []
    missingValues = []

    for line in lines[1:]:
        line = line.strip().split(',')[1:]
        line = [l.strip() for l in line]
        if not line.__contains__(''):  dataSet.append(line)
        else:
            missingValues.append(line)
    dataSet = np.array(dataSet)




    ''' handling missing values '''
    mean = np.array(dataSet[: , -2]).astype(np.float).mean()
    for m in missingValues:
        m[-2] = mean
    dataSet = dataSet.tolist() + missingValues
    dataSet = np.array(dataSet)



    ''' encoding class labels '''
    labels = list(set(dataSet[:, -1]))
    encodeLabel = {}
    i = 0
    for l in labels:
        encodeLabel[l] = i
        i+=1
    labels = [encodeLabel[x] for x in dataSet[:, -1]]


    ''' processing continous attributes '''
    for i in continuousAttributeIndices:
        contData = list(map(float, dataSet[:, i]))
        contDataTuples = []
        for j in range(len(labels)):
            contDataTuples.append((contData[j], labels[j]))
        splitPoint = binarizer(contDataTuples)
        for j in range(len(dataSet)):
            if float(dataSet[j][i]) <= splitPoint: dataSet[j][i] = 0
            else: dataSet[j][i] = 1





    ''' processing labels for continuous attributes '''
    for i in range(len(dataSet)):
        dataSet[i][-1] = encodeLabel[dataSet[i][-1]]


    ''' processing labels for discrete attributes '''
    encodedDisreteAttributes = {}
    for i in range(len(dataSet[0])-1):
        if i not in continuousAttributeIndices:
            encodedDisreteAttributes[i] = {}
            uniqueLabelsInAColumn = list(set(dataSet[:, i]))
            k = 0
            for u in uniqueLabelsInAColumn:
                encodedDisreteAttributes[i][u] = k
                k += 1

            for j in range(len(dataSet)):
                dataSet[j][i] = encodedDisreteAttributes[i][dataSet[j][i]]

    dataSet = dataSet.astype(np.float)
    np.random.shuffle(dataSet)
    return dataSet





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

    dataSet = churnDataset()
    trainSize = int(len(dataSet)*0.8)
    trainSet, testSet = dataSet[:trainSize] , dataSet[trainSize:]


    model = DecisionTree()
    model.train(trainSet)

    print('Dataset : Churn')
    print('----------------\n')






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



    print('---------------Sklearn ------------------------\n\n')

    skModel  = AdaBoostClassifier(n_estimators=50)
    skModel.fit(trainSet[:, :-1], trainSet[:, -1])
    error = 0
    pred = skModel.predict(testSet[:,:-1])
    for i in range(len(testSet)):
        if pred[i] != testSet[i][-1]: error += 1
    acc = (1 - (error/len(testSet)))
    print('Testset Accuracy = ' , acc)










