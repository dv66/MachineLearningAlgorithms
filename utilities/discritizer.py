import numpy as np

'''
    Binarization Using the method mentioned in Slide
'''


def getEntropy(x):
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


def binarizer(X):
    parentLabels = [x[1] for x in X]
    labels = list(set(parentLabels))
    parentEntropy = getEntropy(parentLabels)
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