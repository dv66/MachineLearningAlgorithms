from algorithms.decision_tree.decision_tree import DecisionTree
import numpy as np

class AdaBoostClassifier:

    def __init__(self):
        self.__H = None
        self.__W = None


    def __getParameters(self, nRounds, dataSet):
        nTotal = len(dataSet)
        W = [1/nTotal for x in range(nTotal)]
        H = []
        Z = []
        for k in range(nRounds):
            resampledIndices = np.random.choice(nTotal,nTotal, p=W)
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
            print('round = ' , str(k+1), ' error = ', error)
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












