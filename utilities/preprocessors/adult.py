import numpy as np
from preprocessors.discritizer import binarizer
from algorithms.decision_tree.decision_tree import DecisionTree
from sklearn.tree import DecisionTreeClassifier


globalSplitPoints = []
missingValues = []
def adultDataTrainset():
    # we have to change this two only
    FILE = 'D:/Education/L4T2/0dipto_L4T2/ML Sessional/AdaBoost/datasets/adult_train.txt'
    continuousAttributeIndices = [0,2,4,10,11,12]

    fp = open(FILE)
    lines = fp.readlines()
    dataSet = []
    missingAttributeData = []
    for line in lines:
        line = line.strip().split(',')
        line = [l.strip() for l in line]
        if not line.__contains__('?'):  dataSet.append(line)
        else:
            missingAttributeData.append(line)


    dataSet = np.array(dataSet)


    ''' handling missing attributes'''
    for i in range(len(dataSet[0])-1):
        if i not in continuousAttributeIndices:
            maxAppear = {}
            for x in dataSet[:, i]:
                if x in maxAppear: maxAppear[x] += 1
                else : maxAppear[x] = 1
            maxList = []
            for m in maxAppear:
                maxList.append((maxAppear[m], m))
            maxList = sorted(maxList)[-1]
            for x in missingAttributeData:
                if x[i] == '?' :
                    x[i] = maxList[1]
            missingValues.append(x[i])


    dataSet = np.array(dataSet).tolist() + missingAttributeData
    dataSet = np.array(dataSet)




    '''encoding class labels'''
    labels = list(set(dataSet[:, -1]))
    encodeLabel = {'>50K': 0, '<=50K': 1}
    labels = [encodeLabel[x] for x in dataSet[:, -1]]


    ''' processing continous attributes '''
    for i in continuousAttributeIndices:
        contData = list(map(float, dataSet[:, i]))
        contDataTuples = []
        for j in range(len(labels)):
            contDataTuples.append((contData[j], labels[j]))
        splitPoint = binarizer(contDataTuples)
        globalSplitPoints.append(splitPoint)
        for j in range(len(dataSet)):
            if float(dataSet[j][i]) <= splitPoint: dataSet[j][i] = 0
            else: dataSet[j][i] = 1


            
    ''' processing labels for continuous attributes'''
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
    return dataSet


















'''
    processing testset
'''
def adultDataTestset():
    # we have to change this two only
    FILE = 'D:/Education/L4T2/0dipto_L4T2/ML Sessional/AdaBoost/datasets/adult_test.txt'
    continuousAttributeIndices = [0, 2, 4, 10, 11, 12]

    fp = open(FILE)
    lines = fp.readlines()
    dataSet = []
    missingAttributeData = []
    for line in lines[1:]:
        line = line.strip().split(',')
        line = [l.strip() for l in line]
        if not line.__contains__('?'):  dataSet.append(line)
        else:
            missingAttributeData.append(line)



    dataSet = np.array(dataSet)

    ''' handling missing attributes'''
    k = 0
    for i in range(len(dataSet[0]) - 1):
        if i not in continuousAttributeIndices:
            for x in missingAttributeData:
                if x[i] == '?':
                    x[i] = missingValues[k]
            k+=1

    dataSet = np.array(dataSet).tolist() + missingAttributeData
    dataSet = np.array(dataSet)



    # encoding class labels
    labels = list(set(dataSet[:, -1]))
    encodeLabel = {'>50K.': 0, '<=50K.': 1}
    labels = [encodeLabel[x] for x in dataSet[:, -1]]

    # processing continous attributes
    k = 0
    for i in continuousAttributeIndices:
        splitPoint = globalSplitPoints[k]
        for j in range(len(dataSet)):
            if float(dataSet[j][i]) <= splitPoint:
                dataSet[j][i] = 0
            else:
                dataSet[j][i] = 1
        k+=1

    # processing labels for continuous attributes
    for i in range(len(dataSet)):
        dataSet[i][-1] = encodeLabel[dataSet[i][-1]]

    # processing labels for discrete attributes
    encodedDisreteAttributes = {}
    for i in range(len(dataSet[0]) - 1):
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

    trainSet = adultDataTrainset()
    testSet  = adultDataTestset()

    model = DecisionTree()
    model.train(trainSet)

    print('Dataset : Adult')
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
    print('testset Accuracy = ' , acc)
    getStats(testSet[:,-1], predictedLabels)








    print('---------------Sklearn ------------------------\n\n')

    skModel  = DecisionTreeClassifier()
    skModel.fit(testSet[:, :-1], testSet[:, -1])
    error = 0
    pred = skModel.predict(testSet[:,:-1])
    for i in range(len(testSet)):
        if pred[i] != testSet[i][-1]: error += 1
    acc = (1 - (error/len(testSet)))
    print('testset Accuracy = ' , acc)


    '''
    0 age: continuous.
    2 fnlwgt: continuous.
    4 education-num: continuous.
    10 capital-gain: continuous.
    11 capital-loss: continuous.
    12 hours-per-week: continuous.
    
    1 workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    3 education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    5 marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
    6 occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
    7 relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    8 race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    9 sex: Female, Male.
    13 native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecua
    
    
    LABEL : 2 CLASSES
    '''