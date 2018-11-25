import numpy as np
import random
from copy import deepcopy
from sklearn.model_selection import train_test_split

''' dataset class
    contains subdatasets and sub feature sets in 
    nodes of decision trees'''
class Dataset:
    def __init__(self , matrix , features):
        self.matrix = matrix
        self.features = features

    def __repr__(self):
        return str(self.features) + '\n' + str(self.matrix)



'''
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
'''


''' Decision Tree node class
    Represents a node in the deicision tree'''
class DecisionTreeNode:
    def __init__(self):
        self.name = None
        self.values = None
        self.leafNode = None
        self.dataSet = None
        self.children = {}

    def setParameters(self, name, values, data=None):
        self.name = name
        self.values = values
        self.dataSet = data
        for val in self.values:
            self.children[val] = None

    def setLeafClass(self, label):
        self.leafNode = label

    def addChild(self, val, node):
        self.children[val] = node


    ''' sometimes return random labels in case of multiple options '''
    def getClass(self):
        if type(self.leafNode) == list: return  random.choice(self.leafNode)
        else: return self.leafNode

    def __repr__(self):
        if self.leafNode != None:
            return "leaf_node_class = " + repr(self.leafNode)
        return 'Node('+self.name + ')'






'''
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
'''



''' Decision Tree classifer '''


class DecisionTree:
    def __init__(self, maxDepth=None):
        self.__rootNode = None
        self.__maxDepth = maxDepth

    def __getEntropy(self, x):
        n = len(x)
        count = {}
        for xx in x:
            if xx in count:
                count[xx] += 1
            else:
                count[xx] = 1
        h = 0
        for c in count:
            p = count[c] / n
            h += (-p * np.log2(p))
        return h

    def __getAvgEntropy(self, x):
        cols = len(x[0])
        nSamples = len(x)
        avgEntropies = []
        for i in range(cols - 1):
            dic = {}
            for it in x:
                if it[i] in dic:
                    dic[it[i]].append(it[-1])
                else:
                    dic[it[i]] = [it[-1]]
            expectedEntropy = 0
            for key in dic:
                selectedSamples = dic[key]
                expectedEntropy += ((len(selectedSamples) / nSamples)
                                    * self.__getEntropy(selectedSamples))
            avgEntropies.append(expectedEntropy)
        return avgEntropies

    def __getFeatureToSplitOn(self, x):
        currentEntropy = self.__getEntropy(x[:, -1])
        informationGain = self.__getAvgEntropy(x)
        for i in range(len(informationGain)):
            informationGain[i] = currentEntropy - informationGain[i]
        return np.argmax(informationGain)

    '''
    Binarization Using the method mentioned in Slide
    '''

    def __binarizer(self, X):
        parentLabels = [x[1] for x in X]
        labels = list(set(parentLabels))
        parentEntropy = self.__getEntropy(parentLabels)
        # 0 -> left, 1 ->right
        leftCounter = [[] for x in range(len(labels))]
        rightCounter = [[] for x in range(len(labels))]

        # DP array initialization
        for i in range(len(X)):
            for x in leftCounter:
                x.append(0)
            for x in rightCounter:
                x.append(0)

        # updating counters from left
        for i in range(1, len(X)):
            c = X[i - 1][1]
            leftCounter[c][i] = leftCounter[c][i - 1] + 1
            for j in range(len(leftCounter)):
                if j != c: leftCounter[j][i] = leftCounter[j][i - 1]

        # updating counters from right
        for i in range(len(X) - 2, -1, -1):
            c = X[i + 1][1]
            rightCounter[c][i] = rightCounter[c][i + 1] + 1
            for j in range(len(rightCounter)):
                if j != c: rightCounter[j][i] = rightCounter[j][i + 1]

        averageEntropies = []
        # calculating average entropies for all points
        for i in range(len(leftCounter[0]) - 1):
            leftS = []
            for l in leftCounter:
                leftS.append(l[i + 1])

            rightS = []
            for r in rightCounter:
                rightS.append(r[i])

            nLeft = sum(leftS)
            nRight = sum(rightS)
            nTotal = nLeft + nRight
            leftEntropy = 0
            rightEntropy = 0

            for k in range(len(leftS)):
                leftprob = leftS[k] / nLeft
                rightprob = rightS[k] / nRight
                if leftprob != 0:
                    leftEntropy -= (leftprob * np.log2(leftprob))
                if rightprob != 0:
                    rightEntropy -= (rightprob * np.log2(rightprob))
            avgEntropy = (nLeft / nTotal) * leftEntropy + (nRight / nTotal) * rightEntropy
            averageEntropies.append(avgEntropy)

        gain = [(parentEntropy - x) for x in averageEntropies]
        splitPointIndex = np.argmax(np.array(gain))
        splitPoint = (X[splitPointIndex][0] + X[splitPointIndex + 1][0]) / 2
        return splitPoint

    def __createSubsetDataset(self, x, val, colId):
        mat = []
        selectedFeatures = deepcopy(x.features)
        selectedFeatures.pop(colId)
        for row in x.matrix:
            if row[colId] == val:
                r = np.delete(row, colId)
                mat.append(r)
        return Dataset(np.array(mat), selectedFeatures)

    def __isAllSameLabel(self, x):
        return len(set(x)) == 1

    def __pluralityValue(self, x):
        val = {}
        for v in x:
            if v[-1] not in val:
                val[v[-1]] = 1
            else:
                val[v[-1]] += 1
        val = [(val[v], v) for v in val.keys()]
        val = sorted(val)
        maxClass = val[-1][1]
        maxClasses = []
        i = len(val) - 1
        while val[i][0] == val[-1][0] and i >= 0:
            maxClasses.append(val[i][1])
            i -= 1
        if len(maxClasses) > 1: return maxClasses
        return maxClass

    def __pluralityValueWhenNoExampleLeft(self, x):
        val = {}
        for v in x:
            if v[-1] not in val:
                val[v[-1]] = 1
            else:
                val[v[-1]] += 1
        val = [(val[v], v) for v in val.keys()]
        val = sorted(val)
        maxClass = val[-1][1]
        maxClasses = []
        i = len(val) - 1
        while val[i][0] == val[-1][0] and i >= 0:
            maxClasses.append(val[i][1])
            i -= 1
        if len(maxClasses) > 1:
            return np.random.choice(maxClasses)
        return maxClass

    '''private function to construct the whole decision tree'''

    def __buildDecisionTree(self, data, currentNode, parentNode=None, addingVal=None, curDepth=0, maxDepth=None):
        matrix = data.matrix
        feat = data.features
        # when maxDepth is reached
        if curDepth == maxDepth:
            label = self.__pluralityValue(data.matrix)
            node = DecisionTreeNode()
            node.leafNode = label
            parentNode.addChild(addingVal, node)
            return

        # when no attribute left
        if not feat:
            label = self.__pluralityValue(parentNode.dataSet.matrix)
            node = DecisionTreeNode()
            node.leafNode = label
            parentNode.addChild(addingVal, node)
            return
        # all same labels -- classification done
        elif self.__isAllSameLabel(matrix[:, -1]):
            node = DecisionTreeNode()
            node.leafNode = matrix[0][-1]
            # corner case for all data labeled same in the initial dataset!
            if parentNode == None:
                currentNode.setParameters('0', ['0'])
                addingVal = '0'
                currentNode.addChild(addingVal, node)
            else:
                parentNode.addChild(addingVal, node)
            return

        # go deeper
        else:
            selectedFeature = self.__getFeatureToSplitOn(matrix)
            featureValues = set(matrix[:, selectedFeature])
            currentNode.setParameters(feat[selectedFeature], featureValues, data)
            if parentNode != None: parentNode.addChild(addingVal, currentNode)

            for val in featureValues:
                subset = self.__createSubsetDataset(data, val, selectedFeature)
                newNode = DecisionTreeNode()
                self.__buildDecisionTree(subset, newNode, currentNode, val, curDepth + 1, maxDepth)

    '''private function to print the whole decision tree'''

    def __printDecisionTree(self, node, indent=''):
        # if node == None: return
        print(indent + '|-' + str(node))
        if node.values:
            for child in node.children:
                self.__printDecisionTree(node.children[child], indent + '\t\t')

    def train(self, data):
        features = [str(x) for x in range(len(data[0]) - 1)]
        self.__rootNode = DecisionTreeNode()
        self.__buildDecisionTree(Dataset(np.array(data), features),
                                 self.__rootNode, maxDepth=self.__maxDepth)

    # predicts label for a given sample
    def __getLabel(self, node, sample):
        if node.leafNode != None:
            return node.getClass()
        currentNodeName = node.name
        attributeValue = sample[int(currentNodeName)]
        if type(attributeValue) == str:
            attributeValue = int(attributeValue)
        try:
            nextNode = node.children[attributeValue]
        except:
            predictedLabel = self.__pluralityValueWhenNoExampleLeft(node.dataSet.matrix)
            return predictedLabel
        return self.__getLabel(nextNode, sample)

    # wrapper function for getLabel
    def predict(self, sample):
        label = self.__getLabel(self.__rootNode, sample)
        return label

    # wrapper function for printDecisionTree
    def toDebugString(self):
        self.__printDecisionTree(self.__rootNode)

    def save(self):
        pass

    def load(self):
        pass








'''
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
'''








''' Adaboost classifier '''

class AdaBoostClassifier:

    def __init__(self):
        self.__H = None
        self.__W = None


    def __getParameters(self, nRounds, dataSet):
        nTotal = len(dataSet)
        W = [1/nTotal for x in range(nTotal)]
        ar = [i for i in range(nTotal)]
        H = []
        Z = []
        for k in range(nRounds):
            resampledIndices = np.random.choice(nTotal, nTotal, p=W)
            resampledData = dataSet[ resampledIndices, :]
            weakLearner = DecisionTree(maxDepth=1)
            weakLearner.train(resampledData)
            error = 0
            for i in range(nTotal):
                y = weakLearner.predict(dataSet[i])
                if y != dataSet[i][-1]: error += W[i]
            if error > 0.5:
                print(error , ' at iter = ',  k)
                continue
            print('round =', str(k + 1), ' : error = ', error)
            for i in range(nTotal):
                y = weakLearner.predict(dataSet[i])
                if y == dataSet[i][-1]: W[i] = W[i]*((error)/(1-error))
            errSum  = sum(W)
            W = [w/errSum for w in W]
            H.append(weakLearner)
            Z.append(np.log((1-error)/error))
        self.__H, self.__W = H,W


    def train(self, nRounds, dataSet):
        self.__getParameters(nRounds=nRounds,dataSet=dataSet)

    def predict(self, sample):
        res = 0
        for i in range(len(self.__H)):
            prediction = self.__H[i].predict(sample)
            if prediction == 0 : res += (-1*self.__W[i])
            else : res += (self.__W[i])
        if res > 0 : return 1
        else : return 0








'''
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
'''








class PreprocessedData:
    def __init__(self):
        self.globalSplitPoints = []
        self.missingValues = []

    def reset(self):
        self.globalSplitPoints.clear()
        self.missingValues.clear()


    '''
        Binarization Using the method mentioned in Slide
    '''

    def getEntropy(self, x):
        n = len(x)
        count = {}
        for xx in x:
            if xx in count:
                count[xx] += 1
            else:
                count[xx] = 1
        h = 0
        for c in count:
            p = count[c] / n
            h += (-p * np.log2(p))
        return h

    def binarizer(self, X):
        parentLabels = [x[1] for x in X]
        labels = list(set(parentLabels))
        parentEntropy = self.getEntropy(parentLabels)
        # 0 -> left, 1 ->right
        leftCounter = [[] for x in range(len(labels))]
        rightCounter = [[] for x in range(len(labels))]
        # DP array initialization
        for i in range(len(X)):
            for x in leftCounter:
                x.append(0)
            for x in rightCounter:
                x.append(0)
        # updating counters from left
        for i in range(1, len(X)):
            c = X[i - 1][1]
            leftCounter[c][i] = leftCounter[c][i - 1] + 1
            for j in range(len(leftCounter)):
                if j != c: leftCounter[j][i] = leftCounter[j][i - 1]
        # updating counters from right
        for i in range(len(X) - 2, -1, -1):
            c = X[i + 1][1]
            rightCounter[c][i] = rightCounter[c][i + 1] + 1
            for j in range(len(rightCounter)):
                if j != c: rightCounter[j][i] = rightCounter[j][i + 1]
        averageEntropies = []
        # calculating average entropies for all points
        for i in range(len(leftCounter[0]) - 1):
            leftS = []
            for l in leftCounter:
                leftS.append(l[i + 1])
            rightS = []
            for r in rightCounter:
                rightS.append(r[i])
            nLeft = sum(leftS)
            nRight = sum(rightS)
            nTotal = nLeft + nRight
            leftEntropy = 0
            rightEntropy = 0
            for k in range(len(leftS)):
                leftprob = leftS[k] / nLeft
                rightprob = rightS[k] / nRight
                if leftprob != 0:
                    leftEntropy -= (leftprob * np.log2(leftprob))
                if rightprob != 0:
                    rightEntropy -= (rightprob * np.log2(rightprob))
            avgEntropy = (nLeft / nTotal) * leftEntropy + (nRight / nTotal) * rightEntropy
            averageEntropies.append(avgEntropy)
        gain = [(parentEntropy - x) for x in averageEntropies]
        splitPointIndex = np.argmax(np.array(gain))
        splitPoint = (X[splitPointIndex][0] + X[splitPointIndex + 1][0]) / 2
        return splitPoint




    def churnDataset(self):
        # we have to change this two only
        FILE = 'D:/Education/L4T2/0dipto_L4T2/ML Sessional/AdaBoost/datasets/telco.csv'
        continuousAttributeIndices = [4, 17, 18]

        fp = open(FILE)
        lines = fp.readlines()
        dataSet = []
        missingValues = []

        for line in lines[1:]:
            line = line.strip().split(',')[1:]
            line = [l.strip() for l in line]
            if not line.__contains__(''):
                dataSet.append(line)
            else:
                missingValues.append(line)
        dataSet = np.array(dataSet)

        ''' handling missing values '''
        mean = np.array(dataSet[:, -2]).astype(np.float).mean()
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
            i += 1
        labels = [encodeLabel[x] for x in dataSet[:, -1]]

        ''' processing continous attributes '''
        for i in continuousAttributeIndices:
            contData = list(map(float, dataSet[:, i]))
            contDataTuples = []
            for j in range(len(labels)):
                contDataTuples.append((contData[j], labels[j]))
            splitPoint = self.binarizer(contDataTuples)
            for j in range(len(dataSet)):
                if float(dataSet[j][i]) <= splitPoint:
                    dataSet[j][i] = 0
                else:
                    dataSet[j][i] = 1

        ''' processing labels for continuous attributes '''
        for i in range(len(dataSet)):
            dataSet[i][-1] = encodeLabel[dataSet[i][-1]]

        ''' processing labels for discrete attributes '''
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
        np.random.shuffle(dataSet)
        return dataSet









    def creditDataset(self):
        FILE = 'D:/Education/L4T2/0dipto_L4T2/ML Sessional/AdaBoost/datasets/creditcard.csv'
        fp = open(FILE)
        lines = fp.readlines()
        dataSet = []


        # only +ve samples will be in this list at first
        fraud = []
        for line in lines[1:]:
            line = line.strip().split(',')
            line = line[:-1] + [line[-1][1]]
            line = list(map(float, line))
            if line[-1] == 1:
                fraud.append(line)
            else:
                dataSet.append(line)
        dataSet = np.array(dataSet)
        np.random.shuffle(dataSet)



        for x in dataSet:
            fraud.append(x)

        # return 28000 samples -> 1/10 th of total data
        dataSet = np.array(fraud[:28000])


        # encoding class labels
        labels = list(set(dataSet[:, -1]))
        encodeLabel = {}
        i = 0
        for l in labels:
            encodeLabel[l] = i
            i += 1
        labels = [encodeLabel[x] for x in dataSet[:, -1]]

        # processing continous attributes
        for i in range(len(dataSet[0]) - 1):
            contData = list(map(float, dataSet[:, i]))
            contDataTuples = []
            for j in range(len(labels)):
                contDataTuples.append((contData[j], labels[j]))
            splitPoint = self.binarizer(contDataTuples)
            for j in range(len(dataSet)):
                if float(dataSet[j][i]) <= splitPoint:
                    dataSet[j][i] = 0
                else:
                    dataSet[j][i] = 1
        return dataSet



    def adultDataTrainset(self):
        # we have to change this two only
        FILE = 'D:/Education/L4T2/0dipto_L4T2/ML Sessional/AdaBoost/datasets/adult_train.txt'
        continuousAttributeIndices = [0, 2, 4, 10, 11, 12]

        fp = open(FILE)
        lines = fp.readlines()
        dataSet = []
        missingAttributeData = []
        for line in lines:
            line = line.strip().split(',')
            line = [l.strip() for l in line]
            if not line.__contains__('?'):
                dataSet.append(line)
            else:
                missingAttributeData.append(line)

        dataSet = np.array(dataSet)

        ''' handling missing attributes'''
        for i in range(len(dataSet[0]) - 1):
            if i not in continuousAttributeIndices:
                maxAppear = {}
                for x in dataSet[:, i]:
                    if x in maxAppear:
                        maxAppear[x] += 1
                    else:
                        maxAppear[x] = 1
                maxList = []
                for m in maxAppear:
                    maxList.append((maxAppear[m], m))
                maxList = sorted(maxList)[-1]
                for x in missingAttributeData:
                    if x[i] == '?':
                        x[i] = maxList[1]
                self.missingValues.append(x[i])

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
            splitPoint = self.binarizer(contDataTuples)
            self.globalSplitPoints.append(splitPoint)
            for j in range(len(dataSet)):
                if float(dataSet[j][i]) <= splitPoint:
                    dataSet[j][i] = 0
                else:
                    dataSet[j][i] = 1

        ''' processing labels for continuous attributes'''
        for i in range(len(dataSet)):
            dataSet[i][-1] = encodeLabel[dataSet[i][-1]]

        ''' processing labels for discrete attributes '''
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












    ''' processing testset '''
    def adultDataTestset(self):
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
            if not line.__contains__('?'):
                dataSet.append(line)
            else:
                missingAttributeData.append(line)

        dataSet = np.array(dataSet)

        ''' handling missing attributes'''
        k = 0
        for i in range(len(dataSet[0]) - 1):
            if i not in continuousAttributeIndices:
                for x in missingAttributeData:
                    if x[i] == '?':
                        x[i] = self.missingValues[k]
                k += 1

        dataSet = np.array(dataSet).tolist() + missingAttributeData
        dataSet = np.array(dataSet)

        # encoding class labels
        labels = list(set(dataSet[:, -1]))
        encodeLabel = {'>50K.': 0, '<=50K.': 1}
        labels = [encodeLabel[x] for x in dataSet[:, -1]]

        # processing continous attributes
        k = 0
        for i in continuousAttributeIndices:
            splitPoint = self.globalSplitPoints[k]
            for j in range(len(dataSet)):
                if float(dataSet[j][i]) <= splitPoint:
                    dataSet[j][i] = 0
                else:
                    dataSet[j][i] = 1
            k += 1

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





    def getStats(self, trueLabels, predictedLabels):
        truePositive = sum([trueLabels[i] == predictedLabels[i] == 1 for i in range(len(trueLabels))])
        falseNegatives = sum([trueLabels[i] == 1 and predictedLabels[i] == 0 for i in range(len(trueLabels))])
        trueNegative = sum([trueLabels[i] == predictedLabels[i] == 0 for i in range(len(trueLabels))])
        falsePositives = sum([trueLabels[i] == 0 and predictedLabels[i] == 1 for i in range(len(trueLabels))])
        # truePositiveRate or sensitivity or TPR
        TPR = truePositive / (truePositive + falseNegatives)
        # trueNegativeRate or specificity or TNR
        TNR = trueNegative / (trueNegative + falsePositives)
        # precision or positive predictive value
        PPV = truePositive / (truePositive + falsePositives)
        # false discovery rate (FDR)
        FDR = 1 - PPV
        # F1 score
        F1 = (2 * truePositive) / ((2 * truePositive) + falsePositives + falseNegatives)
        print('truePositiveRate or sensitivity or TPR = ', TPR)
        print('trueNegativeRate or specificity or TNR = ', TNR)
        print('precision or positive predictive value = ', PPV)
        print('false discovery rate (FDR) = ', FDR)
        print('F1 score = ', F1)




    def zipLabelWithFeatures(self, X, y):
        X = X.tolist()
        y = y.tolist()
        for i in range(len(X)):
            X[i] = X[i] + [y[i]]
        return np.array(X)


    def runSimulationChurnDecisionTree(self):
        dataSet = self.churnDataset()
        trainFeat, testFeat , trainLabel , testLabel = train_test_split(dataSet[:, :-1],
                                                                      dataSet[:,-1],
                                                                      test_size=0.2,
                                                                      stratify=dataSet[:,-1])
        trainSet = self.zipLabelWithFeatures(trainFeat, trainLabel)
        testSet = self.zipLabelWithFeatures(testFeat, testLabel)

        model = DecisionTree()
        model.train(trainSet)

        print('\n\n\n\n\nDataset : Churn')
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
        acc = (1 - (error / len(trainSet)))
        print('Trainset Accuracy = ', acc)
        self.getStats(trainSet[:, -1], predictedLabels)

        print('\n\n\n\nTestSet Statistics')
        print('-------------------')
        # on TestSet
        error = 0
        predictedLabels = []
        for s in testSet:
            pred = model.predict(s)
            predictedLabels.append(pred)
            if pred != s[-1]: error += 1
        acc = (1 - (error / len(testSet)))
        print('Testset Accuracy = ', acc)
        self.getStats(testSet[:, -1], predictedLabels)
        print('************************************************************')
        print('************************************************************')
        print('************************************************************\n\n\n')


        self.reset()

    def runSimulationAdultDecisionTree(self):
        trainSet = self.adultDataTrainset()
        testSet = self.adultDataTestset()

        model = DecisionTree()
        model.train(trainSet)

        print('\n\n\n\n\nDataset : Adult')
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
        acc = (1 - (error / len(trainSet)))
        print('Trainset Accuracy = ', acc)
        self.getStats(trainSet[:, -1], predictedLabels)

        print('\n\n\n\nTestSet Statistics')
        print('-------------------')
        # on TestSet
        error = 0
        predictedLabels = []
        for s in testSet:
            pred = model.predict(s)
            predictedLabels.append(pred)
            if pred != s[-1]: error += 1
        acc = (1 - (error / len(testSet)))
        print('testset Accuracy = ', acc)
        self.getStats(testSet[:, -1], predictedLabels)
        print('************************************************************')
        print('************************************************************')
        print('************************************************************\n\n\n')
        self.reset()

    def runSimulationCreditDecisionTree(self):
        dataSet = self.creditDataset()
        trainFeat, testFeat, trainLabel, testLabel = train_test_split(dataSet[:, :-1],
                                                                      dataSet[:, -1],
                                                                      test_size=0.2,
                                                                      stratify=dataSet[:, -1])
        trainSet = self.zipLabelWithFeatures(trainFeat, trainLabel)
        testSet = self.zipLabelWithFeatures(testFeat, testLabel)

        model = DecisionTree()
        model.train(trainSet)

        print('\n\n\n\n\nDataset : Creditcard Fraud')
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
        acc = (1 - (error / len(trainSet)))
        print('Trainset Accuracy = ', acc)
        self.getStats(trainSet[:, -1], predictedLabels)

        print('\n\n\n\nTestSet Statistics')
        print('-------------------')
        # on TestSet
        error = 0
        predictedLabels = []
        for s in testSet:
            pred = model.predict(s)
            predictedLabels.append(pred)
            if pred != s[-1]: error += 1
        acc = (1 - (error / len(testSet)))
        print('Testset Accuracy = ', acc)
        self.getStats(testSet[:, -1], predictedLabels)
        print('************************************************************')
        print('************************************************************')
        print('************************************************************\n\n\n')
        self.reset()


    def runSimulationChurnAdaBoost(self, nRounds):
        dataSet = self.churnDataset()
        trainFeat, testFeat , trainLabel , testLabel = train_test_split(dataSet[:, :-1],
                                                                      dataSet[:,-1],
                                                                      test_size=0.2,
                                                                      stratify=dataSet[:,-1])
        trainSet = self.zipLabelWithFeatures(trainFeat, trainLabel)
        testSet = self.zipLabelWithFeatures(testFeat, testLabel)

        model = AdaBoostClassifier()
        model.train(nRounds, trainSet)

        print('\n\n\n\n\nDataset : Churn')
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
        acc = (1 - (error / len(trainSet)))
        print('Trainset Accuracy = ', acc)
        self.getStats(trainSet[:, -1], predictedLabels)

        print('\n\n\n\nTestSet Statistics')
        print('-------------------')
        # on TestSet
        error = 0
        predictedLabels = []
        for s in testSet:
            pred = model.predict(s)
            predictedLabels.append(pred)
            if pred != s[-1]: error += 1
        acc = (1 - (error / len(testSet)))
        print('Testset Accuracy = ', acc)
        self.getStats(testSet[:, -1], predictedLabels)
        print('************************************************************')
        print('************************************************************')
        print('************************************************************\n\n\n')


        self.reset()

    def runSimulationAdultAdaBoost(self, nRounds):
        trainSet = self.adultDataTrainset()
        testSet = self.adultDataTestset()

        model = AdaBoostClassifier()
        model.train(nRounds, trainSet)

        print('\n\n\n\n\nDataset : Adult')
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
        acc = (1 - (error / len(trainSet)))
        print('Trainset Accuracy = ', acc)
        self.getStats(trainSet[:, -1], predictedLabels)

        print('\n\n\n\nTestSet Statistics')
        print('-------------------')
        # on TestSet
        error = 0
        predictedLabels = []
        for s in testSet:
            pred = model.predict(s)
            predictedLabels.append(pred)
            if pred != s[-1]: error += 1
        acc = (1 - (error / len(testSet)))
        print('testset Accuracy = ', acc)
        self.getStats(testSet[:, -1], predictedLabels)
        print('************************************************************')
        print('************************************************************')
        print('************************************************************\n\n\n')
        self.reset()

    def runSimulationCreditAdaBoost(self, nRounds):
        dataSet = self.creditDataset()
        trainFeat, testFeat, trainLabel, testLabel = train_test_split(dataSet[:, :-1],
                                                                      dataSet[:, -1],
                                                                      test_size=0.2,
                                                                      stratify=dataSet[:, -1])
        trainSet = self.zipLabelWithFeatures(trainFeat, trainLabel)
        testSet = self.zipLabelWithFeatures(testFeat, testLabel)

        model = AdaBoostClassifier()
        model.train(nRounds, trainSet)

        print('\n\n\n\n\nDataset : Creditcard Fraud')
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
        acc = (1 - (error / len(trainSet)))
        print('Trainset Accuracy = ', acc)
        self.getStats(trainSet[:, -1], predictedLabels)

        print('\n\n\n\nTestSet Statistics')
        print('-------------------')
        # on TestSet
        error = 0
        predictedLabels = []
        for s in testSet:
            pred = model.predict(s)
            predictedLabels.append(pred)
            if pred != s[-1]: error += 1
        acc = (1 - (error / len(testSet)))
        print('Testset Accuracy = ', acc)
        self.getStats(testSet[:, -1], predictedLabels)
        print('************************************************************')
        print('************************************************************')
        print('************************************************************\n\n\n')
        self.reset()


'''
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
'''



preprocessedData = PreprocessedData()


# preprocessedData.runSimulationChurnDecisionTree()
# preprocessedData.runSimulationAdultDecisionTree()
# preprocessedData.runSimulationCreditDecisionTree()


# preprocessedData.runSimulationChurnAdaBoost(nRounds=20)
# preprocessedData.runSimulationAdultAdaBoost(nRounds=20)

preprocessedData.runSimulationCreditAdaBoost(nRounds=5)
