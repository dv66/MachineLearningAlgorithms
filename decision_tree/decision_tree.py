from algorithms.decision_tree.dataset import *
from copy import deepcopy
from algorithms.decision_tree.decision_tree_node import DecisionTreeNode

class DecisionTree:
    def __init__(self, maxDepth = None):
        self.__rootNode = None
        self.__maxDepth = maxDepth



    def __getEntropy(self, x):
        n = len(x)
        count = {}
        for xx in x:
            if xx in count:
                count[xx] +=1
            else: count[xx] = 1
        h = 0
        for c in count:
            p = count[c] / n
            h += (-p * np.log2(p))
        return h


    def __getAvgEntropy(self, x):
        cols = len(x[0])
        nSamples = len(x)
        avgEntropies = []
        for i in range(cols-1):
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
        currentEntropy = self.__getEntropy(x[:,-1])
        informationGain = self.__getAvgEntropy(x)
        for i in range(len(informationGain)):
            informationGain[i] = currentEntropy - informationGain[i]
        return np.argmax(informationGain)

    '''
    Binarization Using the method mentioned in Slide
    '''

    def __binarizer(self,X):
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



    def __createSubsetDataset(self, x, val ,colId):
        mat = []
        selectedFeatures = deepcopy(x.features)
        selectedFeatures.pop(colId)
        for row in x.matrix:
            if row[colId] == val:
                r = np.delete(row, colId)
                mat.append(r)
        return Dataset(np.array(mat) , selectedFeatures)


    def __isAllSameLabel(self, x):
        return len(set(x)) == 1


    def __pluralityValue(self, x):
        val = {}
        for v in x:
            if v[-1] not in val : val[v[-1]] = 1
            else : val[v[-1]] += 1
        val = [(val[v], v) for v in val.keys()]
        val = sorted(val)
        maxClass = val[-1][1]
        maxClasses = []
        i = len(val)-1
        while val[i][0] == val[-1][0] and i >= 0:
            maxClasses.append(val[i][1])
            i-=1
        if len(maxClasses) > 1 : return maxClasses
        return maxClass


    def __pluralityValueWhenNoExampleLeft(self, x):
        val = {}
        for v in x:
            if v[-1] not in val : val[v[-1]] = 1
            else : val[v[-1]] += 1
        val = [(val[v], v) for v in val.keys()]
        val = sorted(val)
        maxClass = val[-1][1]
        maxClasses = []
        i = len(val)-1
        while val[i][0] == val[-1][0] and i >= 0:
            maxClasses.append(val[i][1])
            i-=1
        if len(maxClasses) > 1 :
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
        elif self.__isAllSameLabel(matrix[:,-1]) :
            node = DecisionTreeNode()
            node.leafNode = matrix[0][-1]
            # corner case for all data labeled same in the initial dataset!
            if parentNode == None :
                currentNode.setParameters('0', ['0'])
                addingVal = '0'
                currentNode.addChild(addingVal, node)
            else :
                parentNode.addChild(addingVal, node)
            return

        # go deeper
        else:
            selectedFeature = self.__getFeatureToSplitOn(matrix)
            featureValues = set(matrix[:,selectedFeature])
            currentNode.setParameters(feat[selectedFeature] , featureValues, data)
            if parentNode != None: parentNode.addChild(addingVal, currentNode)

            for val in featureValues:
                subset = self.__createSubsetDataset(data ,val, selectedFeature)
                newNode = DecisionTreeNode()
                self.__buildDecisionTree(subset, newNode, currentNode ,val, curDepth+1, maxDepth)





    '''private function to print the whole decision tree'''
    def __printDecisionTree(self, node, indent=''):
        # if node == None: return
        print(indent  +'|-'+ str(node))
        if node.values:
            for child in node.children:
                self.__printDecisionTree(node.children[child], indent + '\t\t')







    def train(self, data):
        features = [str(x) for x in range(len(data[0])-1)]
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
        label = self.__getLabel(self.__rootNode ,sample)
        return label



    # wrapper function for printDecisionTree
    def toDebugString(self):
        self.__printDecisionTree(self.__rootNode)

    def save(self):
        pass

    def load(self):
        pass