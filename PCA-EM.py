import numpy as np
import matplotlib.pyplot as plt


# FILE = '../dataset/data.txt'
FILE = '../dataset/online.txt'
fp = open(FILE)

def normalize(dataSet):
    mean = np.mean(dataSet[:,:-1], axis=0)
    maxElement = np.max(dataSet[:,:-1], axis=0)
    minElement = np.min(dataSet[:, :-1], axis=0)
    dataSet = dataSet.tolist()
    for i in range (len(dataSet)):
        for j in range (len(dataSet[i])-1):
            dataSet[i][j] = (dataSet[i][j]-minElement[j])/(maxElement[j]-minElement[j])
    return np.array(dataSet)

data = []
for line in fp.readlines():
    line= line.strip().split()
    data.append(line)
data = np.array(data).astype(float)
# data = normalize(data)
covarianceMatrix = np.cov(np.transpose(data))

eigenvalues, eigenvectors = np.linalg.eig(covarianceMatrix)
eigenvectors = eigenvectors.T
pc1, pc2 = eigenvectors[0], eigenvectors[1]


X = []
y = []

for d in data:
    pc1Projection = np.dot(d, pc1)
    pc2Projection = np.dot(d, pc2)
    X.append(pc1Projection)
    y.append(pc2Projection)


'''PCA Plot'''
plt.scatter(X, y,c='g',marker='.')
plt.show()




def getProbabilityFromGaussianDensity(x, mean, cov, dimension):
    d = dimension
    prob = 1/np.sqrt(pow(2*np.pi, d) * np.linalg.det(cov))
    temp = np.mat(np.linalg.inv(cov)) * np.transpose(np.mat(x - mean))
    val = np.mat(x-mean) * temp
    return prob * np.exp(-0.5 * np.array(val)[0][0])

def getLabelFromProbability(prob):
    ind = np.argmax(prob)
    return ind




'''
EM Algorithm
in our case
k = 3
'''
K = 4
dimension = 2
N = len(data)
data = np.array([[X[i], y[i]] for i in range(N)])

'''initialization'''
means = np.array([[np.random.uniform(0,1) for j in range(dimension)] for i in range(K)])
covariances = np.array([np.identity(dimension) for i in range(K)])
mixingCoefficients = np.array([1/K for i in range(K)])
probabilities = [[0 for k in range(K)] for i in range(N)]


logLikelihood = []
prev = 0
labels=  None
while True:
    ''' E step'''
    for i in range (N):
        totalProb = 0
        prob = []
        for k in range(K):
            p_i_k = mixingCoefficients[k] * getProbabilityFromGaussianDensity(data[i], means[k], covariances[k],
                                                                              dimension=2)
            prob.append(p_i_k)
            totalProb += p_i_k
        prob = [p/totalProb for p in prob]
        probabilities[i] = prob



    ''' M step'''
    for k in range(K):
        upMean = np.array([0,0]).astype(float)
        upCovariance = np.array([[0, 0],[0, 0]]).astype(float)
        downMean = 0
        for i in range(N):
            downMean += probabilities[i][k]
            upMean += (probabilities[i][k] * data[i])
        means[k] = upMean / downMean
        for i in range(N):
            upCovariance += (probabilities[i][k] * np.transpose(np.mat(data[i])-means[k]) * np.mat(data[i]-means[k]))
        covariances[k] = upCovariance/downMean
        mixingCoefficients[k] = downMean/N



    ''' Evaluate step'''
    curLoglikelihood = 0
    for i in range(N):
        totalProb = 0
        prob = []
        for k in range(K):
            totalProb += mixingCoefficients[k] * getProbabilityFromGaussianDensity(data[i], means[k], covariances[k],
                                                                              dimension=2)
        curLoglikelihood += np.log(totalProb)
    logLikelihood.append(curLoglikelihood)
    cur = curLoglikelihood

    # print('Loglikelihood Value = ', -cur)
    if np.abs(cur - prev) < 0.0001:
        break
    prev = cur

    labels = [getLabelFromProbability(probabilities[i]) for i in range(N)]
    plt.scatter(data[:, 0], data[:, 1],c=labels,marker='.')
    plt.show()






print('# of iterations = ', len(logLikelihood))
# logLikelihood = logLikelihood[1:]
x = [i for i in range(len(logLikelihood))]
plt.plot(x, logLikelihood ,c='g',marker='.')
plt.show()

data = data.tolist()
data = [data[i] + [labels[i]] for i in range(len(data))]
np.savetxt('al.txt', data)