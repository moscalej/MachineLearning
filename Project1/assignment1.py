from scipy.io import loadmat
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

#q1 - K-means
def Kmeans(X,K):
    '''
    :param X: data - np.array with dimension is d X n
    :param K: number of classes
    :return:
            centroids - all k centroids
            C -class of each sample. serial number from centroids
    '''
    print('Run K-means algorithm for K='+str(K)+'...')
    # calc initial centroids
    centroids = np.array([np.random.choice(X[i,:],K) for i in range(X.shape[0])])
    while True:
        # clustering
        C=np.array([np.argmin([np.linalg.norm(x.reshape(1,X.shape[0])-yk) for yk in centroids.T])for x in X.T])
        # move centroid step
        lastCentroids = centroids
        centroids = np.array([X[:,C==k].mean(axis=1) for k in range(K)]).T
        if np.array_equal(lastCentroids,centroids):
            break
    return centroids,C

def PCA(newDim,X):
    '''
    reduce dimension of X from dXn to kXn
    :param newDim: desired dimension
    :param X: data dXn, n- num of samples. d- attributes
    :return Z: X projected on K dimensions
    '''
    print('Calculate PCA...')
    meanX = np.mean(X,axis=1,keepdims=True)
    zeroMeanX = X-meanX
    cov = np.cov(zeroMeanX)
    eigenValues,eigenVectors = LA.eigh(cov)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    eigenVectorsNewDim = eigenVectors[:,0:newDim]
    Z = np.dot(eigenVectorsNewDim.T,X)
    return Z

def PlotClusteringAndRealLabels(Z,C,K,y):
    '''
    plot graph of K-means & real labels
    :param Z: data 2Xn array
    :param C: labels of each sample by clustering k-means
    :param K: number of classes in C
    :param y: real labels of each sample
    '''
    def PlotSingleGRaph(Z,labels,title,K):
        '''
        :param Z:data 2Xn array
        :param labels: labels of each sample
        :param title: for graph
        :param K: number of classes
        :return:
        '''
        for k in range(K):
            plt.scatter(Z[0,labels==k],Z[1,labels==k],label = 'class='+str(k))
        plt.title(title)
        plt.legend()
    print('Plot Graphs...')
    plt.figure()
    plt.subplot(1, 2, 1)
    PlotSingleGRaph(Z, C, 'K-means result, K='+str(K),K)
    plt.subplot(1,2,2)
    PlotSingleGRaph(Z, y, 'real labels',2)

def main():
    data = loadmat('BreastCancerData.mat')
    X = data['X']
    Y = data['y']
    y = np.array([elem[0] for elem in Y])

    # assign q1 - K-means
    # q1.1+q1.2
    K = 2
    centroids2, C2 = Kmeans(X, K)
    # q1.3
    Z = PCA(2, X)
    PlotClusteringAndRealLabels(Z, C2, K, y)
    # q1.q4
    K = 3
    centroids3, C3 = Kmeans(X, K)
    PlotClusteringAndRealLabels(Z, C3, K, y)
    K = 4
    centroids4, C4 = Kmeans(X, K)
    PlotClusteringAndRealLabels(Z, C4, K, y)
    K = 5
    centroids5, C5 = Kmeans(X, K)
    PlotClusteringAndRealLabels(Z, C5, K, y)
    plt.show(block=False)
    print('finish')
if __name__=='__main__':
    main()
