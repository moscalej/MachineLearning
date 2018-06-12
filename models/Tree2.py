import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
# from conda._vendor.tqdm._tqdm import tqdm
from sklearn.model_selection import KFold
from math import log2
import random

def los_fact(criretion):
    def calc_entropy(y):
        # Entropy calculation
        if y.shape[0] == 0:
            return 0
        y0 = float(np.sum(y == 0)) / y.shape[0]
        y1 = float(np.sum(y == 1)) / y.shape[0]
        if y0==0:
            res0=0
        else:
            res0=-y0*log2(y0)
        if y1==0:
            res1=0
        else:
            res1=-y1*log2(y1)
        return res0+res1

    def calc_gini_index( y):
        # Gini index calcualtion
        if y.shape[0] == 0:
            return 0
        y0 = float(np.sum(y == 0)) / y.shape[0]
        y1 = float(np.sum(y == 1)) / y.shape[0]
        return y0 * (1 - y0) + y1 * (1 - y1)

    def calc_classification_error(y):
        # Classification error calculation
        if y.shape[0] == 0:
            return 0
        y0 = float(np.sum(y == 0)) / y.shape[0]
        y1 = float(np.sum(y == 1)) / y.shape[0]
        return 1 - max(y0, y1)
    return dict(ClassificationError= calc_classification_error,
                GiniIndex=calc_gini_index ,
                Enthropy=calc_entropy )[criretion]


class Node:
    def __init__(self, value, feature, node_type, chields=[], balance=[0,0]):
        self.feature = feature
        self.value=value
        self.node_type=node_type
        self.chields=chields
        self.balance = balance

    def predict(self,x):
        if self.node_type == 'Error':
            raise Exception('We made a error node need to de-bug')
        if self.node_type == 'leaf':
            return self.value
        elif x[self.feature] <= self.value:
            return self.chields[0].predict(x)
        else:
            return self.chields[1].predict(x)

    def __str__(self):
        return f'Type: {self.node_type} , value: {round(self.value,2)},' \
               f' feature: {self.feature}, balance: {self.balance}'


class Tree:
    def __init__(self, depht, loss,dropping = True):
        self.dropping = dropping
        self.depht = depht
        self._test = los_fact(loss)
        self.root = None

    def fit(self, X, Y):
        """
        This method will generate a binary tree where each node will ask a relevant question
        update the root of the tree
        :param X:
        :param Y:
        :return:
        """
        self.root = self._split_tree(X, Y, 0)

    def predict(self,x):
        # y_hat = pd.Series(index=x.index)
        y_hat = x.apply(self.root.predict,axis=1)
        return y_hat

    def score(self,x,y):
        return np.mean(self.predict(x) == y)

    def _min_loss_cut(self, feature, Y):
        '''
        this function return the ent and the thresold (desire decision question)
        :param feature: all samples with specific feature data
        :param Y: label of those samples
        :return:
        '''
        classes = Y.unique()
        # fea_order = feature.sort_values().unique()
        treshold = np.linspace(np.min(feature), np.max(feature), np.min([100,y.size]))
        ent,value = 999 ,999
        best_th = np.nan
        for th in treshold:
            # Usin th, split the input data
            left_x = feature[feature <= th]
            left_y = Y[left_x.index]

            right_x = feature[feature > th]
            right_y = Y[right_x.index]

            left_ent = self._test(left_y)
            right_ent = self._test(right_y)

            this_ent = (left_x.shape[0] * left_ent + right_x.shape[0] * right_ent) \
                       / (left_x.shape[0] + right_x.shape[0])

            if this_ent < ent:
                ent = this_ent
                value = th
        return {'score':ent ,'value':value}


    def _find_cut(self, X, Y):
        """
        this function return the ent and the thresold (desire decision question) for the desired feature with minimum of ent
        :param X:columns are the features and rows are the samples. NXD
        :param Y: labels of the samples.
        :return:result: dictionary. key='score',value = ent; key='feature',value = desired feature
        {'score':gini ,'criterion':'le' or 'g','value': value ,'feature': wich feature should cut}
        """
        result = self._min_loss_cut(X[0], Y)
        # for feature in tqdm(X.columns):
        for feature in X.columns:
            feature_r  = self._min_loss_cut(X[feature], Y)
            if feature_r['score'] <= result['score']:
                result = feature_r
                result['feature']= feature
        return result

    def _split_tree(self, X, Y, depht):
        '''
        This method will generate a binary tree where each node will ask a relevant question
        :param X:data. columns are the features and rows are the samples. NXD
        :param Y: labels of the samples.
        :param depht:
        :return:
        '''
        if len(X) == 0:
            raise Exception(f'We build and error node on the fit with depht {depht} and y:{Y_labels.shape}')
        elif len(Y.unique()) != 2:
            node = Node(Y.values[0], 'leaf', 'leaf', [],[sum(Y==0),sum(Y==1)])
            return node
        elif self.depht == depht or X.shape[1] == 1:
            return Node(np.argmax([np.sum(Y == 0), np.sum(Y == 1)]), 'leaf', 'leaf', [],[sum(Y==0),sum(Y==1)])
        info = self._find_cut(X, Y)

        feature = info['feature']
        value = info['value']
        left_x = X[X[feature] <= value]
        if self.dropping == True:
            #left_x =left_x.drop(columns=feature)
            left_x =left_x.drop(feature,axis=1)

        left_y = Y[left_x.index]
        right_x = X[X[feature] > value]
        if self.dropping == True:
            #right_x = right_x.drop(columns=feature)
            right_x = right_x.drop(feature,axis=1)
        right_y =Y[right_x.index]

        if len(left_y) == 0 :
            return Node(np.argmax([np.sum(right_y == 0), np.sum(right_y == 1)]),
                        'leaf', 'leaf', [], [sum(right_y == 0), sum(right_y == 1)])
        if len(right_y) == 0 :
            return Node(np.argmax([np.sum(left_y == 0), np.sum(left_y == 1)]),
                        'leaf', 'leaf', [], [sum(left_y == 0), sum(left_y == 1)])




        node_left = self._split_tree(left_x, left_y, depht + 1)
        node_right= self._split_tree(right_x, right_y, depht + 1)
        node = Node(value, feature, 'Desition', [node_left, node_right],[left_y.size,right_y.size])
        return node


    def predict(self, x):
        # y_hat = pd.Series(index=x.index)
        y_hat = x.apply(self.root.predict, axis=1)
        return y_hat

    def score(self, x, y):
        return np.mean(self.predict(x) == y)



def GetDataFromMatlab(matFile):
    '''
    :param matFile: .m file
    :return: X-data np.array ,y-labels np.array
    '''
    data = loadmat(matFile)
    X = np.array(data['X']).T
    Y = data['y']
    y = np.array([elem[0] for elem in Y])
    return X,y

def SplitDataToTrainAndTestSet(X,y):
    '''
    mix the indexes and choose training and test set in random choice
    :param X:data. NXD
    :param y: labels
    :return: training and test set
    '''
    N = X.shape[0]
    idxs = np.linspace(0, N - 1, N)
    # random.shuffle(idxs)
    idxs_training_set = idxs[0:int(0.8 * N)]
    idxs_test_set = idxs[int(0.8 * N):]

    X_training_set = np.array([X[int(i),:] for i in idxs_training_set])
    X_test_set = np.array([X[int(i),:] for i in idxs_test_set])
    y_training_set = np.array([y[int(i)] for i in idxs_training_set])
    y_test_set = np.array([y[int(i)] for i in idxs_test_set])
    return X_training_set,X_test_set,y_training_set,y_test_set

def GetTrainAndTestForCV(Train_x,Train_y,i):
    valSize = int(0.1*Train_x.shape[0])
    val_x = pd.DataFrame(Train_x[i*valSize:(i+1)*valSize,:])
    val_y = pd.Series(Train_y[i*valSize:(i+1)*valSize].reshape(-1))
    train_x = pd.DataFrame(np.concatenate((Train_x[:i*valSize,:], Train_x[(i+1)*valSize:,:]), axis=0))
    train_y = pd.Series(np.concatenate((Train_y[:i*valSize], Train_y[(i+1)*valSize:]), axis=0).reshape(-1))
    return train_x,val_x,train_y,val_y

def PlotResultCV(err1, err2, err3):
    x = [1, 2, 3]
    y = np.array(err1['mean'], err2['mean'], err3['mean'])
    e = np.array(err1['std'], err2['std'], err3['std'])
    plt.errorbar(x, y, e, linestyle='None', marker='^')

    plt.xlabel('x:1=GiniIndex,2=ClassificationError,3=Enthropy')
    plt.ylabel('error')
    plt.title('error for each model via cross-validation')
    plt.show()



if __name__ == '__main__':
    # from sklearn.model_selection import train_test_split
    X, Y = GetDataFromMatlab('BreastCancerData.mat')
    Train_x, Test_x, Train_y, Test_y = SplitDataToTrainAndTestSet(X, Y)
    # a = loadmat(r'D:\Ale\Documents\Technion\ML\MachineLearning\Data\BreastCancerData.mat',appendmat=False)
    a = loadmat(r'BreastCancerData.mat', appendmat=False)
    x = pd.DataFrame(a['X'].T)
    y = pd.Series(a['y'].reshape(-1))
    # results = pd.DataFrame(columns = ['GiniIndex','ClassificationError', 'Enthropy'], index= [10,50,99])
    results = pd.DataFrame(columns=['ClassificationError', 'Enthropy','GiniIndex'], index=[10, 50, 99])
    error = {}
    for col in results.columns:
        error[col] = {}
        error[col]['val'] = []
        print('check col ' + col)
        for cv in range(10):
            print('check cv=' + str(cv))
            # row=10
            # ctrain_x , ctest_x, ctrain_y, ctest_y = train_test_split(x,y,train_size=row,test_size=300)
            train_x, val_x, train_y, val_y = GetTrainAndTestForCV(Train_x, Train_y, cv)
            tree = Tree(10, col)
            tree.fit(train_x, train_y)
            results.loc[cv, col] = 1 - tree.score(val_x, val_y)
            print('error = ' + str(results.loc[cv, col]))
            error[col]['val'].append(results.loc[cv, col])
        error[col]['mean'] = np.mean(error[col]['val'])
        error[col]['std'] = np.std(error[col]['val'])

    PlotResultCV(error['GiniIndex'], error['ClassificationError'], error['Enthropy'])

    bestClassifier = results.columns[np.argmax([error[col]['mean'] for col in results.columns])]
    print('best classifier is ' + bestClassifier)
    # train on all data
    trainTree = Tree(10, bestClassifier)
    trainTree.fit(train_x, train_y)
    valErr = 1 - trainTree.score(train_x, train_y)
    print('error on validaion set is: ' + str(valErr))
    # test set
    testTree = Tree(10, bestClassifier)
    testTree.fit(train_x, train_y)
    testErr = 1 - trainTree.score(train_x, train_y)
    print('error on validaion set is: ' + str(testErr))

    print('finish!')


