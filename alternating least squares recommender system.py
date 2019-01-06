import numpy as np
from copy import deepcopy
import  matplotlib.pyplot as plt



def trainTestValidateSplit(d):
    trainSet = 99 * np.ones((len(d),len(d[0])))
    testSet = 99 * np.ones((len(d),len(d[0])))
    validationSet = 99 * np.ones((len(d),len(d[0])))

    for i in range(len(d)):
        totKnown = []
        for j in range(len(d[i])):
            if d[i][j] != 99:
                totKnown.append(j)
        lenToKnown = len(totKnown)
        lenTrain = int(lenToKnown * 0.6)
        lenTest = int(lenToKnown*0.2)
        lenValidate = lenToKnown-lenTrain-lenTest
        np.random.shuffle(totKnown)

        trainIndices = totKnown[:lenTrain]
        testIndices = totKnown[lenTrain:lenTrain+lenTest]
        validationIndices = totKnown[lenTrain+lenTest:]
        for ind in trainIndices:
            trainSet[i][ind] = d[i][ind]
        for ind in testIndices:
            testSet[i][ind] = d[i][ind]
        for ind in validationIndices:
            validationSet[i][ind] = d[i][ind]

    return np.array(trainSet),np.array(testSet),np.array(validationSet)



def getRMSE(trn, tst):
    nCnt = 0
    tot = 0
    for i in range(len(trn)):
        for j in range(len(trn[0])):
            if trn[i][j] != 99 and tst[i][j] != 99:
                tot += ((trn[i][j]-tst[i][j]) ** 2)
                nCnt += 1
    return (tot/nCnt) ** 0.5



def ALS(train,test, K, regu):
    '''
    N = # of rows in rating matrix
    M = # of cols in rating matrix
    K: latent factor dimension
    '''
    N = len(train)
    M = len(train[0])
    K = K

    U = np.random.uniform(0,1,(N,K))
    V = np.zeros((K,M))

    cnt = 0
    regularizationParameterV = regu
    regularizationParameterU = regu
    plotY = []
    finalError = None
    model = None
    prev = np.inf



    for e in range(10):
        for m in range(M):
            val1 = np.zeros((K,K))
            for n in range(N):
                if train[n][m] != 99:
                    val1 += np.mat(U[n]).T * np.mat(U[n])
            val1 += (regularizationParameterV * np.identity(K))
            val1 = np.linalg.inv(val1)
            val2 = np.zeros((1,K))
            for n in range(N):
                if train[n][m] != 99:
                    val2 += train[n][m] * np.mat(U[n])
            V[:,m] = np.matmul(val1,val2[0])
        for n in range(N):
            val1 = np.zeros((K,K))
            for m in range(M):
                if train[n][m] != 99:
                    val1 += np.mat(V[:,m]).T * np.mat(V[:,m])
            val1 += (regularizationParameterU * np.identity(K))
            val1 = np.linalg.inv(val1)
            val2 = np.zeros((1,K))
            for m in range(M):
                if train[n][m] != 99:
                    val2 += train[n][m] * np.mat(V[:,m])
            U[n] = np.matmul(val1, val2[0])

        res = np.mat(U) * np.mat(V)
        firstTerm = 0
        for i in range(len(train)):
            for j in range(len(train[0])):
                if train[i][j] != 99:
                    val = (train[i][j] - np.array(np.mat(U[i]) * np.mat(V[:, j]).T)[0][0])
                    firstTerm += val*val

        secondTerm = 0
        for n in range(N):
            norm2= np.linalg.norm(U[n])
            secondTerm += (norm2*norm2)
        secondTerm *= regularizationParameterU

        thirdTerm = 0
        for m in range(M):
            norm2 = np.linalg.norm(V[:,m])
            thirdTerm += (norm2*norm2)
        thirdTerm*= regularizationParameterV

        lReg = firstTerm + secondTerm + thirdTerm
        errorRMSE = getRMSE(np.array(res), test)
        finalError = errorRMSE


        ''' if convex point found'''
        if finalError > prev:
            break
        prev = finalError
        model = np.array(res)
        # print(errorRMSE)
        plotY.append(errorRMSE)
        # print(lReg)
        # plotY.append(lReg)


    # plotX = np.arange(0,len(plotY))
    # plt.plot(plotX,plotY)
    # plt.show()
    # print('--------------------------------------------------------------------------')

    return finalError, model

'''
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
'''









def tuning(trn, val):
    '''
    Hyperparameter Tuning
    '''

    Ks= [5,10,20,40]
    regularizationConstants = [0.01, 0.1, 1.0, 10.0]

    bestModel = None
    bestError = np.inf
    bestK = None
    bestLambda = None
    for k in Ks:
        for l in regularizationConstants:
            err, model = ALS(trn,val,k,l)
            if err < bestError:
                bestError = err
                bestModel = model
                bestK = k
                bestLambda = l
            print('error at K = ', k, ' lambda = ', l, ' -- ',bestError)

    print(bestModel)
    print(bestError)
    print('chosen hyperparameters : K =',bestK,' , lambda = ' ,bestLambda)
    print('========================================================================')
    return bestModel






def recommenderEngine(alsModel, testSet):
    err = getRMSE(alsModel, testSet)
    print('Testset Error = ', err)




if __name__ == '__main__':
    FILE = 'D:/Education/L4T2/0dipto_L4T2/ML Sessional/Assignment3/RecommenderEngine/datasets/data.txt'
    fp = open(FILE)
    data = np.array([line.strip().split(',')[1:] for line in fp.readlines()]).astype(float)

    SIZEx = 300
    SIZEy = 40
    data = data[:SIZEx, :SIZEy]

    train, test, validate = trainTestValidateSplit(data)

    bestModel = tuning(train,validate)

    recommenderEngine(bestModel, test)







